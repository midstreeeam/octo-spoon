"""Inference helpers for the Octodiff diffusion language model."""

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
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from octodiff.config import OctodiffConfig
from octodiff.model import OctodiffForDiffusionLM

_LOGGER = logging.getLogger(__name__)


def _load_config(config_arg: Optional[str]) -> tuple[OctodiffConfig, Optional[Path]]:
    if config_arg is None:
        return OctodiffConfig(), None

    path = Path(config_arg)
    if path.is_file() and path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return OctodiffConfig(**payload), path.parent
    if path.exists():
        config = OctodiffConfig.from_pretrained(str(path))
        return config, path if path.is_dir() else path.parent

    config = OctodiffConfig.from_pretrained(config_arg)
    return config, None


def _load_state_dict(checkpoint_path: Path) -> dict:
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install safetensors to load .safetensors checkpoints.") from exc
        return load_safetensors(str(checkpoint_path))

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    return payload.get("model_state", payload)


def _load_tokenizer(
    tokenizer_arg: Optional[str],
    config_dir: Optional[Path],
    checkpoint_dir: Optional[Path],
) -> AutoTokenizer:
    candidates: list[str] = []
    if tokenizer_arg:
        candidates.append(tokenizer_arg)
    for option in (config_dir, checkpoint_dir, _SCRIPT_DIR):
        if option is not None:
            candidates.append(str(option))

    last_error: Optional[Exception] = None
    for location in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(location)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
            return tokenizer
        except (OSError, ValueError) as exc:
            last_error = exc
            continue

    raise ValueError("Unable to load tokenizer; provide --tokenizer or colocate files with the checkpoint.") from last_error


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


@torch.no_grad()
def generate(
    model: OctodiffForDiffusionLM,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generated_ids = model.sample(
        input_ids,
        attention_mask=attention_mask,
        num_steps=num_steps,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        guidance_scale=guidance_scale,
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


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
        vocab = _infer_vocab_size(state_dict)
        if vocab is not None and vocab != config.vocab_size:
            config.vocab_size = vocab

    model = OctodiffForDiffusionLM(config).to(device)
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _LOGGER.warning("Missing keys when loading checkpoint: %s", missing)
        if unexpected:
            _LOGGER.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    tokenizer = _load_tokenizer(args.tokenizer, config_dir, checkpoint_dir)
    return generate(
        model,
        tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )


def _infer_vocab_size(state_dict: dict) -> Optional[int]:
    for key in ("lm_head.weight", "model.embed_tokens.weight"):
        tensor = state_dict.get(key)
        if tensor is not None:
            return int(tensor.shape[0])
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Octodiff text generation")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to condition the model")
    parser.add_argument("--config", type=str, default=None, help="Path or name of the OctodiffConfig")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint containing model weights")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path or name")
    parser.add_argument("--device", type=str, default=None, help="Computation device (e.g. cuda, cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of new tokens to sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Diffusion denoising steps (defaults to config.denoise_steps)",
    )
    parser.add_argument("--guidance-scale", type=float, default=None, help="Classifier-free guidance scale")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    text = run_inference(args)
    print(text)


if __name__ == "__main__":
    main()
