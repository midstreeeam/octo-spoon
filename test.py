import torch

from torchinfo import summary

from octo_model.config import OctoConfig
from octo_model.model import OctoForCausalLM

from octodiff.config import OctodiffConfig
from octodiff.model import OctodiffForDiffusionLM


def summarize_octo_model() -> None:
    config = OctoConfig(pad_token_id=0)
    model = OctoForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 16))

    print("=== OctoForCausalLM ===")
    model_summary = summary(
        model,
        input_data={"input_ids": input_ids},
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        depth=4,
        verbose=0,
    )
    print(model_summary)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    print("Logits shape:", outputs["logits"].shape)


def summarize_octodiff_model() -> None:
    config = OctodiffConfig(pad_token_id=0)
    model = OctodiffForDiffusionLM(config)

    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    timesteps = torch.rand(batch_size)
    noise = torch.randn(batch_size, seq_len, config.hidden_size)

    print("=== OctodiffForDiffusionLM ===")
    diff_summary = summary(
        model,
        input_data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "timesteps": timesteps,
            "noise": noise,
        },
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        depth=4,
        verbose=0,
    )
    print(diff_summary)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            timesteps=timesteps,
            noise=noise,
        )
    print("Diffusion logits shape:", outputs["logits"].shape)
    print("Diffusion loss:", float(outputs["loss"]))


if __name__ == "__main__":
    # summarize_octo_model()
    summarize_octodiff_model()
