import argparse
import os

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
)

from hausa_lm.data import get_dataset
from hausa_lm.trainer.arguments import HausaLMTrainingArgs

#!/usr/bin/env python3


def register_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal language model and push to HF Hub"
    )
    parser.add_argument(
        "--source_model_id",
        type=str,
        required=True,
        help="HF hub ID of the model to fine-tune",
    )
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default=None,
        help="HF hub ID of the tokenizer to use (if different from model)",
    )
    parser.add_argument(
        "--target_model_id",
        type=str,
        required=True,
        help="HF hub ID to push the fine-tuned model to",
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
        "--max_seq_length",
        type=int,
        default=512,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--features", nargs="+", default=["text"], help="Dataset features to use"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading models from the Hub",
    )

    # Add new arguments to override training parameters
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Override the number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Override batch size for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=None,
        help="Override batch size for evaluation",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Override optimizer (e.g., adamw_torch, adamw_hf)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )

    return parser.parse_args()


def main():
    args = register_args()

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

    # Update training args with command line parameters
    training_args.features = args.features
    training_args.hub_model_id = args.target_model_id

    # Overriding training arguments with command line parameters
    if args.num_train_epochs is not None:
        training_args.num_train_epochs = args.num_train_epochs
    if args.per_device_train_batch_size is not None:
        training_args.per_device_train_batch_size = args.per_device_train_batch_size
    if args.per_device_eval_batch_size is not None:
        training_args.per_device_eval_batch_size = args.per_device_eval_batch_size
    if args.optim is not None:
        training_args.optim = args.optim
    if args.gradient_accumulation_steps is not None:
        training_args.gradient_accumulation_steps = args.gradient_accumulation_steps

    # Determine tokenizer ID (use model ID if not specified)
    tokenizer_id = args.tokenizer_id or args.source_model_id

    print(f"Downloading tokenizer from {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, token=hf_token, trust_remote_code=args.trust_remote_code
    )

    print(f"Loading model configuration from {args.source_model_id}")
    # Load config from pretrained model but initialize model with random weights
    config = AutoConfig.from_pretrained(
        args.source_model_id,
        vocab_size=tokenizer.vocab_size,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=args.trust_remote_code
    )
    print("Initialized model with random weights")

    print("Loading training dataset")
    train_dataset = get_dataset(
        hub_url=training_args.dataset_url,
        subset=training_args.train_subset,
        features=training_args.features,
        max_seq_length=args.max_seq_length,
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
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        split=training_args.eval_split,
        n_samples=training_args.eval_samples,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    training_args.hub_token = hf_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training")
    trainer.train()

    print(f"Pushing model to Hugging Face Hub: {args.target_model_id}")
    trainer.save_model()
    trainer.push_to_hub(token=hf_token)

    print(f"✅ Model successfully fine-tuned and pushed to {args.target_model_id}")


if __name__ == "__main__":
    main()
