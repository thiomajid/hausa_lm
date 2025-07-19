import argparse
from pathlib import Path

import yaml
from transformers import AutoConfig


def export_config(model_hub_id: str, output_dir: str, token: str = None):
    """
    Export the configuration of a model to a YAML file.

    Args:
        model_hub_id (str): The Hugging Face model hub ID (e.g., 'microsoft/DialoGPT-medium').
        output_dir (str): The directory to save the configuration file.
        token (str, optional): The token to use for authentication.
    """

    try:
        # Load configuration using AutoConfig (works with any model)
        config = AutoConfig.from_pretrained(model_hub_id, token=token)

        # Create output directory if it doesn't exist
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Convert config to dictionary
        config_dict = config.to_dict()

        # Generate filename from model hub ID
        model_name = model_hub_id.replace("/", "_")
        yaml_file_path = out_dir / f"{model_name}.yaml"

        # Save as YAML
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(config_dict, yaml_file, default_flow_style=False)

        print(f"Configuration for {model_hub_id} saved to {yaml_file_path}")

    except Exception as e:
        print(f"Error downloading config for {model_hub_id}: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export model configuration to YAML file."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The Hugging Face model hub ID (e.g., 'microsoft/DialoGPT-medium').",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/model",
        help="The directory to save the configuration file (default: configs/model).",
    )

    parser.add_argument(
        "--token",
        type=str,
        help="The Hugging Face token to use for authentication.",
    )

    args = parser.parse_args()

    # Convert relative path to absolute path
    if not Path(args.output_dir).is_absolute():
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / args.output_dir
    else:
        output_dir = args.output_dir

    export_config(model_hub_id=args.model, output_dir=output_dir, token=args.token)
