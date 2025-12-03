# ReconST Package

Optimal Gene Panel Selection for Targeted Spatial Transcriptomics Experiments

## Installation

```bash
pip install -e .
```

## Data Structure

The package expects data to be organized in the following structure:

```
./example_data/
  ├── example_sc.h5ad
  ├── example_merfish.h5ad
  ├── example_merfish_cell_metadata.csv
  ├── example_merfish_ccf_coordinates.csv
  └── example_merfish_gene.csv

./gene_panels/
  └── all_gene_symbols.csv (optional, can be generated)

./results/
  └── (output files will be saved here)
```

## Usage

See `example.ipynb` for a complete example.

```python
import reconst
from reconst import FeatureScreeningAutoencoder, prepare_common_genes, create_data_loader
from reconst import train_model, evaluate_model, select_genes
```
