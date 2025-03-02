from transformers import AutoTokenizer, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from hausa_lm.modules import HausaLMForCausalLM

from .arguments import HausaLMTrainingArgs


class HausaLMTrainer(Trainer):
    def __init__(
        self,
        model: HausaLMForCausalLM,
        args: HausaLMTrainingArgs,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        super().__init__(model=model, args=args, processing_class=tokenizer, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        outputs: CausalLMOutputWithPast = model(**inputs)
        loss = outputs.loss
        perplexity = loss.exp().item()

        metrics = {
            "ce_loss": loss.item(),
            "perplexity": perplexity,
        }

        self.log(metrics)

        if return_outputs:
            return (loss, outputs)

        return loss
