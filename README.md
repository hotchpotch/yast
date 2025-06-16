# YAST - Yet Another SPLADE or Sparse Trainer 🚀

Welcome to YAST! This open-source project provides a powerful and flexible SPLADE (Sparse Lexical and Expansion) trainer. Built to integrate seamlessly with Huggingface's Trainer API, YAST allows you to leverage cutting-edge sparse retrieval techniques based on various SPLADE-related research papers. Our goal is to offer an accessible tool for training these models. YAST is licensed under the permissive MIT License.

## ⚠️ Important Notice

Please note that YAST is currently an **experimental** project. This means you might encounter **breaking changes** introduced from time to time. To ensure a stable experience, we highly recommend **forking** this repository and working with a specific **revision (commit hash)**.

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires Python 3.11.

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/hotchpotch/yast.git
cd yast

# Create virtual environment and install dependencies
uv venv --python 3.11 .venv
uv sync --extra dev

# Activate virtual environment (optional - you can use uv run instead)
source .venv/bin/activate

# Run training example
uv run python -m yast.run examples/japanese-splade/toy.yaml
```

### Optional: Flash Attention 2 for Performance

For improved training speed, you can install Flash Attention 2:

```bash
# Install Flash Attention 2 (optional, for performance optimization)
uv pip install --no-deps flash-attn --no-build-isolation
```

**Note**: Flash Attention 2 requires a compatible CUDA GPU and may take some time to compile.

## Training a Japanese SPLADE Model

For details on training a Japanese SPLADE model, please see the [Japanese SPLADE example](./examples/japanese-splade/README.md). This document is written in Japanese (日本語で書かれています). If you don't read Japanese, online translation tools can be helpful for understanding the content.

### Related Blog Posts (Content in Japanese)

Here are some blog posts related to this project, written in Japanese:
- [高性能な日本語SPLADE（スパース検索）モデルを公開しました](https://secon.dev/entry/2024/10/07/100000/)
- [SPLADE モデルの作り方・日本語SPLADEテクニカルレポート](https://secon.dev/entry/2024/10/23/080000-japanese-splade-tech-report/)
- [情報検索モデルで最高性能(512トークン以下)・日本語版SPLADE v2をリリース](https://secon.dev/entry/2024/12/19/100000-japanese-splade-v2-release/)


## 💡 Related Work

Another project, [YASEM (Yet Another Splade | Sparse Embedder)](https://github.com/hotchpotch/yasem), offers a more user-friendly implementation for working with SPLADE models.

## 🙏 Acknowledgments

We thank the researchers behind the original SPLADE papers for their outstanding contributions to this field.

## References

- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)  
- [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://arxiv.org/abs/2109.10086)  
- [From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective](http://arxiv.org/abs/2205.04733)  
- [An Efficiency Study for SPLADE Models](https://dl.acm.org/doi/10.1145/3477495.3531833)  
- [A Static Pruning Study on Sparse Neural Retrievers](https://arxiv.org/abs/2304.12702)  
- [SPLADE-v3: New baselines for SPLADE](https://arxiv.org/abs/2403.06789)  
- [Minimizing FLOPs to Learn Efficient Sparse Representations](https://arxiv.org/abs/2004.05665)  

## License

This project is licensed under the MIT License. See the LICENSE file for full license details.  
Copyright (c) 2024 Yuichi Tateno (@hotchpotch)
