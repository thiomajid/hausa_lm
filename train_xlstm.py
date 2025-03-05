import argparse
import os

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)

from hausa_lm.config import HausaLMConfig
from hausa_lm.data import get_dataset
from hausa_lm.modules import HausaLMForCausalLM
from hausa_lm.trainer.arguments import HausaLMTrainingArgs
from hausa_lm.trainer.trainer import HausaLMTrainer

#!/usr/bin/env python3


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a HausaLM xLSTM model and push to HF Hub"
    )

    parser.add_argument(
        "--tokenizer_id",
        type=str,
        required=True,
        help="HF hub ID of the tokenizer to use",
    )
    parser.add_argument(
        "--target_model_id",
        type=str,
        required=True,
        help="HF hub ID to push the trained model to",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for pushing the model",
    )
    parser.add_argument(
        "--training_args_file",
        type=str,
        default=None,
        help="Path to a Yaml file with training arguments",
    )

    parser.add_argument(
        "--features", nargs="+", default=["text"], help="Dataset features to use"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading models from the Hub",
    )

    args = parser.parse_args()

    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token is required. Provide it with --hf_token or set HF_TOKEN environment variable"
        )

    # Parse training arguments
    if args.training_args_file:
        training_args = HfArgumentParser(HausaLMTrainingArgs).parse_yaml_file(
            args.training_args_file
        )[0]
    else:
        # Use default training arguments
        training_args = HausaLMTrainingArgs()

    # Update training args
    training_args.features = args.features
    training_args.hub_token = hf_token
    training_args.push_to_hub = True
    training_args.hub_model_id = args.target_model_id

    print(f"Downloading tokenizer from {args.tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id, token=hf_token, trust_remote_code=args.trust_remote_code
    )

    print(f"Loading model configuration from {training_args.xlstm_config_path}")
    config = HausaLMConfig.from_yaml(training_args.xlstm_config_path)

    print("Initializing HausaLM model")
    model = HausaLMForCausalLM(config=config)
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters"
    )

    print("Loading training dataset")
    train_dataset = get_dataset(
        hub_url=training_args.dataset_url,
        subset=training_args.train_subset,
        features=training_args.features,
        max_seq_length=config.text_config.context_length,
        tokenizer=tokenizer,
        split=training_args.train_split,
        n_samples=training_args.train_samples,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    print("Loading evaluation dataset")
    eval_dataset = get_dataset(
        hub_url=training_args.dataset_url,
        subset=training_args.eval_subset,
        features=training_args.features,
        max_seq_length=config.text_config.context_length,
        tokenizer=tokenizer,
        split=training_args.eval_split,
        n_samples=training_args.eval_samples,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    # Initialize the HausaLM trainer
    trainer = HausaLMTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training")
    trainer.train()

    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    print(f"Pushing model to Hugging Face Hub: {training_args.hub_model_id}")
    trainer.push_to_hub(
        repo_id=args.target_model_id,
        token=hf_token,
    )

    print(f"✅ HausaLM model successfully trained and pushed to {args.target_model_id}")


if __name__ == "__main__":
    main()
