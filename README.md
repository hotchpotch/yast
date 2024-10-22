# YAST - Yet Another SPLADE or Sparse Trainer

This is an open-source implementation of a SPLADE trainer, designed to work with Huggingface's Trainer API by referencing various SPLADE-related papers. The project is licensed under the permissive MIT License.

## Important Notice

This repository is **experimental**, meaning that **breaking changes** may be introduced frequently. When using this project, it is recommended to **fork** the repository and work with a specific **revision** to ensure stability.

## Training a Japanese SPLADE Model

See the [Japanese SPLADE example](./examples/japanese-splade-base-v1/README.md) for details.

## Related Work

We also provide another project, [YASEM (Yet Another Splade | Sparse Embedder)](https://github.com/hotchpotch/yasem), which offers a more user-friendly implementation for working with SPLADE models.

## Acknowledgments

We would like to express our gratitude to the researchers behind the original SPLADE papers for their outstanding contributions to this field.

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