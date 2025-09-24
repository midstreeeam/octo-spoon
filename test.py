import torch

from octo_model.config import OctoConfig
from octo_model.model import OctoForCausalLM

from torchinfo import summary



if __name__ == "__main__":
    config = OctoConfig(pad_token_id=0)
    model = OctoForCausalLM(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 16))

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
