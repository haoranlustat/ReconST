# ReconST: Optimal Gene Panel Selection for Targeted Spatial Transcriptomics Experiments

![Graphical Abstract](Graphical%20Abstract.png)

## 1. Introduction

ReconST is a data-driven framework for designing optimal gene panels for targeted spatial transcriptomics experiments. Modern spatial transcriptomics platforms such as MERFISH, seqFISH, Xenium, and MERSCOPE can only measure a limited number of genes, making the choice of panel crucial for capturing transcriptomic structure and spatial organization. ReconST addresses this challenge by learning which genes best preserve global expression patterns and biological variation when compressed to a small panel.

The method uses a gated autoencoder trained on scRNA-seq data to identify the most informative genes for reconstructing the full transcriptome. The resulting gene panel preserves cell-type structure and spatial patterns when transferred to spatial datasets, as demonstrated using a high-resolution mouse brain MERFISH atlas. ReconST is implemented as a lightweight Python package with a simple, reproducible workflow for model training, gene-ranking, and exporting panels compatible with modern spatial transcriptomics platforms.

### Key Features

- End-to-end gene selection using a gated autoencoder  
- L1-based sparsity for compact and interpretable gene panels  
- Directly operates on scRNA-seq data  
- Lightweight and easy-to-use Python API  

### Citation

[Lu, H., et al. *"Optimal Gene Panel Selection for Targeted Spatial Transcriptomics Experiments."* bioRxiv (2025): 2025-10.](https://www.biorxiv.org/content/10.1101/2025.10.08.681071v1.abstract)



## 2. Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/haoranlustat/ReconST.git
```



## 3. Quick Start

```python
import reconst
from reconst import (
    FeatureScreeningAutoencoder,
    prepare_common_genes,
    create_data_loader,
    train_model,
    evaluate_model,
    select_genes,
)

# 1) Prepare shared gene set
common_genes = prepare_common_genes(sc_adata, merfish_adata)

# 2) Build dataloader
loader = create_data_loader(sc_adata[:, common_genes], batch_size=256)

# 3) Train gated autoencoder
model = FeatureScreeningAutoencoder(n_genes=len(common_genes))
train_model(model, loader, n_epochs=1000)

# 4) Select top genes
selected_genes = select_genes(model, top_k=200)

# 5) Optional: evaluate on spatial data
metrics = evaluate_model(model, merfish_adata[:, common_genes])
```


## Example

A complete working example is provided here:

**[example.ipynb](example.ipynb)**




## 4. Method Overview

ReconST uses a gated autoencoder architecture to identify genes that best reconstruct the full transcriptome when compressed to a small panel. A learnable gating layer assigns an importance weight to each gene, and an L1 sparsity penalty encourages most gates to approach zero so that the model focuses on a compact, informative subset of genes.

During training, the gated expression matrix is passed through an encoder–decoder network that learns a low-dimensional representation and reconstructs the original expression profile. Genes with consistently high gate values are considered informative, while those with near-zero weights are excluded. After convergence, the final gene panel is obtained from the non-zero gates or by selecting the top-ranked genes. This end-to-end formulation provides a simple and scalable way to learn biologically meaningful gene panels suitable for targeted spatial transcriptomics.


## 5. Documentation

Full documentation and tutorials:

https://haoranlustat.github.io/ReconST/
