# ReconST: Optimal Gene Panel Selection for Targeted Spatial Transcriptomics Experiments



# 1.Introduction

ReconST is a Python package for automated and data-driven gene panel design in targeted spatial transcriptomics experiments.
It uses a gated autoencoder to identify the most informative subset of genes for reconstructing transcriptomic structure, enabling efficient and biologically meaningful panel selection.


## Citation:
Lu, Haoran, et al. "Optimal Gene Panel Selection for Targeted Spatial Transcriptomics Experiments." bioRxiv (2025): 2025-10.
https://www.biorxiv.org/content/10.1101/2025.10.08.681071v1.abstract



## Key Features

- End-to-end gene selection using a gated autoencoder  
- L1-based sparsity for compact and interpretable gene panels  
- Directly operates on scRNA-seq data  
- Produces gene lists compatible with MERFISH, seqFISH, Xenium, MERSCOPE, etc.  
- Lightweight, minimal-dependency Python API  

-------------------------------------------------------------------------------

# 2.Installation

Install directly from GitHub:

pip install git+https://github.com/haoranlustat/ReconST.git



## Quick Start

import reconst
from reconst import (
    FeatureScreeningAutoencoder,
    prepare_common_genes,
    create_data_loader,
    train_model,
    evaluate_model,
    select_genes,
)

## 1. Prepare shared gene set
common_genes = prepare_common_genes(sc_adata, merfish_adata)

## 2. Build dataloader
loader = create_data_loader(sc_adata[:, common_genes], batch_size=256)

## 3. Train gated autoencoder
model = FeatureScreeningAutoencoder(n_genes=len(common_genes))
train_model(model, loader, n_epochs=1000)

## 4. Select top genes
selected_genes = select_genes(model, top_k=200)

## 5. Optional: evaluate on spatial data
metrics = evaluate_model(model, merfish_adata[:, common_genes])

-------------------------------------------------------------------------------

Method Overview

ReconST introduces a learnable gating layer that assigns an importance weight to each gene.
L1 regularization produces sparse selections, and genes with non-zero gate values form the final panel.
This approach integrates gene scoring and representation learning into a simple, unified training pipeline.

-------------------------------------------------------------------------------

Documentation

Full documentation and tutorials:
https://haoranlustat.github.io/ReconST/
