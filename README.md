# Hausa Language Models

## Overview

This repository contains research and implementations of language models for Hausa, one of Africa's major languages spoken primarily in northern Nigeria, Niger, and other parts of West Africa. The project aims to advance natural language processing capabilities for Hausa by developing specialized language models for various tasks.

## Objectives

- Develop pre-trained language models for Hausa
- Fine-tune models for specific NLP tasks
- Improve accessibility of NLP tools for the Hausa-speaking community
- Bridge the technological gap for low-resource languages

## Tasks

The models in this repository are being developed for various NLP/Vision tasks, including but not limited to:

- Text generation
- Machine translation
- Vision question answering

## Motivation

Despite being spoken by over 70 million people, Hausa remains underrepresented in current NLP research and applications. This project aims to address this disparity by creating resources that can be used in practical applications and further research.

## Contributing

Contributions are welcome! Whether you're a native Hausa speaker, ML practitioner, or NLP researcher, your input can help improve these resources.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Scripts

Here you can include various scripts related to the development and fine-tuning of the Hausa language models. This may include pre-processing scripts, training scripts, and any utility functions that are helpful for working with the data or models.

- To train a custom tokenizer

```bash
python3 train_tokenizer.py \
    --base_tokenizer "<BASE_TOKENIZER_NAME>" \
    --dataset_url "<DATASET_NAME>" \
    --subset "<SUBSET_NAME>" \
    --split "<SPLIT_NAME>" \
    --text_column "<TEXT_COLUMN>" \
    --trust_remote_code \
    --push_to_hub \
    --model_id "<YOUR_MODEL_ID>" \
    --token "<YOUR_HF_TOKEN>"
```

- To train an xLSTM causal language model

```bash
python3 train_xlstm.py \
    --target_model_id "<YOUR_MODEL_ID>" \
    --tokenizer_id "<TOKENIZER_ID>" \
    --hf_token "<YOUR_HF_TOKEN>" \
    --features "<TEXT_COLUMN>" \
    --training_args_file "<PATH_TO_TRAINING_ARGS>" \
    --trust_remote_code
```

- To finetune a model of your choosing already available on Hugging Face

```bash
python3 finetune_hf.py \
    --source_model_id "<SOURCE_MODEL_ID>" \
    --target_model_id "<YOUR_MODEL_ID>" \
    --tokenizer_id "<TOKENIZER_ID>" \
    --hf_token "<YOUR_HF_TOKEN>" \
    --max_seq_length 512 \
    --features "<TEXT_COLUMN>" \
    --training_args_file "<PATH_TO_TRAINING_ARGS>" \
    --trust_remote_code
```

## Citation

If you use these models in your research or applications, please cite this repository as

```tex
@misc{thiombiano2024hausa_lm,
    author = {Thiombiano, Abdoul Majid O.},
    title = {Hausa Language Models},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/thiomajid/hausa_lm}}
}
```
