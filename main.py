import torch

from hausa_lm import HausaLMConfig, HausaLMModelForCausalLM
from hausa_lm.utils import model_summary

if __name__ == "__main__":
    # parser = HfArgumentParser(HausaLMTrainingArgs)
    # args = parser.parse_yaml_file("./trainer_arguments.yaml")[0]
    # args = cast(HausaLMTrainingArgs, args)

    config = HausaLMConfig.from_yaml("./model_config.yaml")
    model = HausaLMModelForCausalLM(config)

    model_summary(model)

    dummy_input = torch.randint(
        1, 100, (2, config.xlstm_config.context_length), device=model.device
    )

    outputs = model(dummy_input)
    print(outputs.logits.shape)
