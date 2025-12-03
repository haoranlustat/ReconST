"""Data preprocessing utilities"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class GeneExpressionDataset(Dataset):
    """Dataset for gene expression data"""

    def __init__(self, matrix):
        self.matrix = matrix

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        return self.matrix[idx]


def prepare_common_genes(adata1, adata2):
    """Find common genes between two datasets and filter both"""
    genes1 = set(adata1.var_names)
    genes2 = set(adata2.var_names)
    common_genes = list(genes1.intersection(genes2))

    genes_to_keep1 = [gene in common_genes for gene in adata1.var_names]
    genes_to_keep2 = [gene in common_genes for gene in adata2.var_names]

    adata1_common = adata1[:, genes_to_keep1].copy()
    adata2_common = adata2[:, genes_to_keep2].copy()

    # Align gene order
    adata2_common = adata2_common[:, adata1_common.var_names].copy()

    return adata1_common, adata2_common, common_genes


def create_data_loader(adata, batch_size=256, train_split=0.8, shuffle=True):
    """Create train and test data loaders from AnnData object"""
    matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_matrix = torch.tensor(matrix, dtype=torch.float32)
    dataset = GeneExpressionDataset(gene_matrix)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
