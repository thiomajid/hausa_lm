import argparse
import os

from peft import LoraConfig, get_peft_model
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
from hausa_lm.utils import dataset_filters_registry

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
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use 16-bit floating point precision for training",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Disable 16-bit floating point precision for training",
    )

    parser.add_argument(
        "--lora",
        action="store_true",
        default=False,
        help="Use LoRA for training",
    )

    parser.add_argument(
        "--no-lora",
        action="store_false",
        dest="lora",
        help="Disable LoRA for training",
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank",
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha",
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.06,
        help="LoRA dropout rate",
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=500,
        help="Number of steps between logging",
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Number of steps between saving checkpoints",
    )

    parser.add_argument("--dataset-url", type=str, help="Dataset URL")
    parser.add_argument(
        "--eval-dataset-url",
        type=str,
        help="Evaluation dataset URL",
    )

    parser.add_argument(
        "--train-subset",
        type=str,
        help="Subset of the training split to use",
    )
    parser.add_argument("--train-split", type=str, help="The split to use for training")
    parser.add_argument("--train-samples", type=int, help="Number of training samples")

    parser.add_argument(
        "--eval-subset",
        type=str,
        help="Subset of the evaluation split to use",
    )

    parser.add_argument(
        "--eval-split",
        type=str,
        help="The split to use for evaluation",
    )

    parser.add_argument("--eval-samples", type=int, help="Number of evaluation samples")

    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        help="List of features to use from the dataset (e.g. --features text title)",
    )

    # Add dataset cache parameters after other dataset controls
    parser.add_argument(
        "--use-dataset-cache",
        action="store_true",
        dest="use_dataset_cache",
        default=True,
        help="Use cached datasets if available",
    )

    parser.add_argument(
        "--no-dataset-cache",
        action="store_false",
        dest="use_dataset_cache",
        help="Don't use cached datasets",
    )

    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default="./.dataset_cache",
        help="Directory to store cached datasets",
    )

    # model architecture specific arguments
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=768,
        help="Hidden size of the model",
    )

    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=12,
        help="Number of hidden layers in the model",
    )

    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=12,
        help="Number of attention heads in the model",
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduling",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
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
    args_dict = vars(args)
    for key, value in args_dict.items():
        if hasattr(training_args, key) and value is not None:
            setattr(training_args, key, value)

    training_args.logging_steps = args.logging_steps
    training_args.save_steps = args.save_steps
    training_args.train_subset = args.train_subset
    training_args.eval_subset = args.eval_subset

    # training_args.features = args.features
    # training_args.hub_model_id = args.target_model_id
    # training_args.hub_token = hf_token
    # training_args.fp16 = args.fp16

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

    # Set model architecture parameters
    for key, value in args_dict.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)

    print(f"Loading model from {args.source_model_id}")
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=args.trust_remote_code,
    )
    print("Initialized model with random weights")

    if args.lora:
        print("Applying LoRA configuration")

        lora_config = LoraConfig(
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    # Display model total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params / 1e6:.2f}M")

    filters = dataset_filters_registry.get(args.dataset_url, None)

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
        use_cache=args.use_dataset_cache,
        cache_dir=args.dataset_cache_dir,
        filters=filters,
    )

    print("Loading evaluation dataset")

    if args.eval_dataset_url is None:
        args.eval_dataset_url = args.dataset_url

    eval_dataset = get_dataset(
        hub_url=args.eval_dataset_url,
        subset=training_args.eval_subset,
        features=training_args.features,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        split=training_args.eval_split,
        n_samples=training_args.eval_samples,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
        use_cache=args.use_dataset_cache,
        cache_dir=args.dataset_cache_dir,
        filters=filters,
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
