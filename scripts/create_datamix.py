import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.mixture import DataMixConfig, MixDatasetConfig, push_datamix_to_hub


@hydra.main(config_path="../configs", config_name="data_mix_config", version_base="1.2")
def main(cfg: DictConfig):
    try:
        print("Starting script...")
        print("parsing config dict")
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        print("Config dict created successfully")

        config = DataMixConfig(**config_dict)
        config.data_mix = [MixDatasetConfig(**el) for el in config.data_mix]
        print("config loaded")

        push_datamix_to_hub(config)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
