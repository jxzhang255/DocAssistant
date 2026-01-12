# DocAssistant: A Multimodal Model for Document Understanding

<div align="center">

[![EMNLP 2025 Findings](https://img.shields.io/badge/EMNLP%202025-Findings-red)](https://aclanthology.org/2025.findings-emnlp.187/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-emnlp.187.pdf)

</div>

## Introduction

This repository contains the official implementation of the EMNLP 2025 Findings paper: **"DocAssistant: Integrating Key-region Reading and Step-wise Reasoning for Robust Document Visual Question Answering"**.

**DocAssistant** is a powerful multimodal model designed for document understanding tasks, including DocVQA, ChartQA, and InfographicsVQA. It builds upon the InternVL architecture to achieve state-of-the-art performance in processing visually rich documents.

> **Note**: Please refer to the [Paper PDF](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-emnlp.187.pdf) for detailed methodology and experimental results.

## News 🚀

- **[2025/11]** DocAssistant is accepted to **EMNLP 2025 Findings**! 🎉

## Installation

Please follow the instructions below to set up the environment.

```bash
conda create -n docassistant python=3.9 -y
conda activate docassistant
pip install -r requirements.txt
```

For more detailed installation steps, please refer to [INSTALLATION.md](./INSTALLATION.md).

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
