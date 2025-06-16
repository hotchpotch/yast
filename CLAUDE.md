# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YAST (Yet Another SPLADE or Sparse Trainer) is an experimental implementation of a SPLADE trainer that works with Huggingface's Trainer API. The project focuses on training sparse neural information retrieval models, particularly for Japanese language applications.

**Important**: This is an experimental repository with frequent breaking changes. Code should be treated accordingly.

## Development Environment

This project uses **uv** for dependency management and requires **Python 3.11**. The project was migrated from Poetry to uv for faster dependency resolution and improved development experience.

## Architecture

### Core Components

- **Training Entry Point**: `yast/run.py` - Main training script that accepts YAML/JSON configuration files
- **Model Architecture**: `yast/modeling/` - Contains SPLADE model implementations:
  - `splade.py` - Base SPLADE model
  - `splade_subword.py` - Subword-aware SPLADE variant
- **Training Logic**: `yast/trainer.py` - Custom trainer extending HuggingFace Trainer
- **Data Pipeline**: `yast/data.py` - Dataset creation and collation logic
- **Custom Datasets**: `yast/custom_dataset/` - Domain-specific dataset implementations

### Configuration System

Training is driven by YAML configuration files that specify:
- Model parameters (ModelArguments)
- Data parameters (DataArguments) 
- Training parameters (SpladeTrainingArguments extending HuggingFace TrainingArguments)
- Run parameters (RunArguments)

Examples are in `examples/japanese-splade/` directory.

## Common Commands

### Environment Setup
```bash
# Initial setup
uv venv --python 3.11 .venv  # Create virtual environment
uv sync --extra dev           # Install dependencies with dev extras

# Daily development
source .venv/bin/activate     # Activate virtual environment (optional)
# OR use uv run for direct command execution without activation
```

### Training
```bash
# Train with YAML config
uv run python -m yast.run path/to/config.yaml

# Train with JSON config  
uv run python -m yast.run path/to/config.json
```

### Code Quality
```bash
uv run ruff check .     # Lint code
uv run ruff format .    # Format code
```

### Package Management
```bash
uv add package_name           # Add new dependency
uv add --dev package_name     # Add development dependency
uv remove package_name        # Remove dependency
uv sync                       # Sync dependencies with lockfile
uv lock                       # Update lockfile
```

### Development Dependencies
- `yasem>=0.3.1` - Related SPLADE embedder project
- `fugashi` + `unidic-lite` - Japanese text processing

## Key Configuration Parameters

- `sparsity_weight_doc/query`: Controls sparsity regularization
- `regularizer_doc/query`: Type of regularization (L1, L2, flops, etc.)
- `training_losses`: Can be single loss or dict with multiple losses and weights
- `subword_pooling`: Enables subword-aware model variant
- `trust_remote_code`: Required for some model loading scenarios

## Testing

No formal test suite is currently implemented. The project uses example configurations for validation.