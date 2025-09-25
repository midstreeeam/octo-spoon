"""Inference helpers for the Octo causal language model.

The module intentionally keeps the entry points simple:

* ``generate`` – feed a prompt, receive generated text.
* ``run_inference``/CLI – convenience wrapper that loads config, tokenizer,
  and weights before calling ``generate``.

It can be executed either with ``python -m octo_model.inference`` or directly as
``python octo_model/inference.py``.  In the latter case we extend ``sys.path`` to
make relative imports work.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
if __package__ is None or __package__ == "":  # Allows ``python octo_model/inference.py``.
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from octo_model.config import OctoConfig
from octo_model.model import OctoForCausalLM

_LOGGER = logging.getLogger(__name__)


def _load_config(config_arg: Optional[str]) -> tuple[OctoConfig, Optional[Path]]:
    if config_arg is None:
        config = OctoConfig()
        return config, None

    path = Path(config_arg)
    if path.is_file() and path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = OctoConfig(**payload)
        return config, path.parent
    if path.exists():
        config = OctoConfig.from_pretrained(str(path))
        return config, path if path.is_dir() else path.parent

    config = OctoConfig.from_pretrained(config_arg)
    return config, None


def _load_state_dict(checkpoint_path: Path) -> dict:
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install safetensors to load .safetensors checkpoints, or convert the file to .pt/.bin."
            ) from exc
        return load_safetensors(str(checkpoint_path))

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    return payload.get("model_state", payload)


def _infer_vocab_size(state_dict: dict) -> Optional[int]:
    for key in ("model.embed_tokens.weight", "embed_tokens.weight", "lm_head.weight"):
        tensor = state_dict.get(key)
        if tensor is not None:
            return int(tensor.shape[0])
    return None


def _load_tokenizer(
    tokenizer_arg: Optional[str],
    config_dir: Optional[Path],
    checkpoint_dir: Optional[Path],
) -> AutoTokenizer:
    candidates = []
    if tokenizer_arg:
        candidates.append(tokenizer_arg)
    for path in (config_dir, checkpoint_dir, _SCRIPT_DIR):
        if path is not None:
            candidates.append(str(path))

    last_error: Optional[Exception] = None
    for target in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(target)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
            return tokenizer
        except (OSError, ValueError) as exc:
            last_error = exc
            continue

    raise ValueError(
        "Failed to load a tokenizer. Provide --tokenizer or place tokenizer files next "
        "to the checkpoint/config."
    ) from last_error


@torch.no_grad()
def generate(
    model: OctoForCausalLM,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    do_sample: bool = False,
    eos_token_id: Optional[int] = None,
) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generated = input_ids
    stop_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated, attention_mask=attention_mask)
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            logits = next_token_logits / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)

            if top_k > 0:
                k = min(top_k, probs.size(-1))
                topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
                filtered = torch.zeros_like(probs).scatter_(-1, topk_indices, topk_probs)
                probs = filtered / filtered.sum(dim=-1, keepdim=True)

            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative <= top_p
                mask[..., 0] = True
                sorted_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                choice = torch.multinomial(sorted_probs, num_samples=1)
                next_token = torch.gather(sorted_indices, -1, choice)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        if attention_mask is not None:
            pad = torch.ones_like(next_token)
            attention_mask = torch.cat([attention_mask, pad], dim=-1)

        if stop_id is not None and int(next_token.item()) == int(stop_id):
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def _resolve_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda")


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_inference(args: argparse.Namespace) -> str:
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    checkpoint_path: Optional[Path] = None
    state_dict: Optional[dict] = None
    checkpoint_dir: Optional[Path] = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        if checkpoint_path.is_dir():
            raise ValueError("Pass the checkpoint file, not a directory.")
        state_dict = _load_state_dict(checkpoint_path)
        checkpoint_dir = checkpoint_path.parent

    config, config_dir = _load_config(args.config)

    if state_dict is not None:
        vocab_size = _infer_vocab_size(state_dict)
        if vocab_size is not None and config.vocab_size != vocab_size:
            _LOGGER.info("Adjusting config vocab_size from %d to %d based on checkpoint.", config.vocab_size, vocab_size)
            config.vocab_size = vocab_size

    if config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id
    config._name_or_path = str(config_dir or checkpoint_dir or _SCRIPT_DIR)

    model = OctoForCausalLM(config).to(device)

    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _LOGGER.warning("Missing parameters when loading weights: %s", ", ".join(missing))
        if unexpected:
            _LOGGER.warning("Unexpected parameters when loading weights: %s", ", ".join(unexpected))

    tokenizer = _load_tokenizer(args.tokenizer, config_dir, checkpoint_dir)

    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        eos_token_id=args.eos_token_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run text generation with the Octo model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint file (.pt/.bin/.safetensors).")
    parser.add_argument("--config", help="Optional path/identifier for OctoConfig. Defaults to built-in config.")
    parser.add_argument("--tokenizer", help="Tokenizer identifier/path if not colocated with the checkpoint.")
    parser.add_argument("--prompt", required=True, help="Prompt text to feed into the model.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of tokens to generate beyond the prompt.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature when sampling.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling cutoff. 1.0 disables the filter.")
    parser.add_argument("--top-k", type=int, default=0, help="Restrict sampling to the top-k tokens when > 0.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--device", help="Torch device specifier, e.g. 'cuda' or 'cpu'. Defaults to auto-detect.")
    parser.add_argument("--seed", type=int, help="Optional random seed for reproducible sampling.")
    parser.add_argument(
        "--eos-token-id",
        type=int,
        help="Override the tokenizer EOS token id used to stop generation.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    text = run_inference(args)
    print(text)


if __name__ == "__main__":  # pragma: no cover
    main()
