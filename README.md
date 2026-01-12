# DocAssistant: A Multimodal Model for Document Understanding

<div align="center">

[![EMNLP 2025 Findings](https://img.shields.io/badge/EMNLP%202025-Findings-red)](https://aclanthology.org/2025.findings-emnlp.187/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-emnlp.187.pdf)

</div>

## Introduction

This repository contains the official implementation of the EMNLP 2025 Findings paper: **"[Please Insert Paper Title Here]"**.

**DocAssistant** is a powerful multimodal model designed for document understanding tasks, including DocVQA, ChartQA, and InfographicsVQA. It builds upon the InternVL architecture to achieve state-of-the-art performance in processing visually rich documents.

> **Note**: Please refer to the [Paper PDF](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-emnlp.187.pdf) for detailed methodology and experimental results.

## News 🚀

- **[2025/11]** DocAssistant is accepted to **EMNLP 2025 Findings**! 🎉
- **[2025/11]** Code and models are released.

## Installation

Please follow the instructions below to set up the environment.

```bash
conda create -n docassistant python=3.9 -y
conda activate docassistant
pip install -r requirements.txt
```

For more detailed installation steps, please refer to [INSTALLATION.md](./INSTALLATION.md).

## Model Zoo

| Model | Description | Download |
| :--- | :--- | :--- |
| DocAssistant-Base | Base model for document understanding | [Coming Soon] |
| DocAssistant-Chat | Chat-tuned model for interactive QA | [Coming Soon] |

## Quick Start

### Inference

You can use the following script to run inference on your own images:

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model_path = "path/to/docassistant/model"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

image = Image.open("examples/image1.jpg").convert('RGB')
pixel_values = ... # Add preprocessing here based on your specific implementation

response = model.chat(tokenizer, pixel_values, "Describe this document.")
print(response)
```

## Evaluation

We provide scripts to reproduce the results on DocVQA, ChartQA, and other benchmarks.

```bash
# Evaluate on DocVQA
bash evaluate.sh docvqa
```

## Citation

If you find this project useful in your research, please cite our paper:

```bibtex
@inproceedings{docassistant2025emnlp,
    title = "{Please Insert Paper Title Here}",
    author = "{Please Insert Authors Here}",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.187",
}
```

## Acknowledgement

This project is built upon [InternVL](https://github.com/OpenGVLab/InternVL). We thank the authors for their open-source contribution.
