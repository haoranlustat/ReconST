"""Training and evaluation functions"""

import torch
from torch import nn


def train_model(model, train_loader, test_loader, num_epochs=20, lr=1e-3,
                weight_decay=1e-5, l_lambda=1e-4, device='cpu'):
    """Train the autoencoder model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for data in train_loader:
            genes = data.to(device)
            screened_features, _, outputs = model(genes)

            loss = criterion(outputs, genes)
            l1_penalty = l_lambda * torch.norm(model.feature_importance, p=1)
            total_loss = loss + l1_penalty

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += loss.item() * genes.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluate on test data
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                genes = data.to(device)
                _, _, outputs = model(genes)
                loss = criterion(outputs, genes)
                test_loss += loss.item() * genes.size(0)

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train: {train_loss:.4f}, Test: {test_loss:.4f}')

    return train_losses, test_losses


def evaluate_model(model, test_loader, gene_mask=None, device='cpu'):
    """Evaluate model on test data, optionally with gene filtering"""
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data in test_loader:
            genes = data.to(device)

            if gene_mask is not None:
                if not isinstance(gene_mask, torch.Tensor):
                    gene_mask = torch.tensor(gene_mask)
                gene_mask = gene_mask.to(device)
                filtered_genes = genes * gene_mask
            else:
                filtered_genes = genes

            _, _, outputs = model(filtered_genes)
            loss = criterion(outputs, genes)
            test_loss += loss.item() * genes.size(0)

    test_loss /= len(test_loader.dataset)
    return test_loss


def select_genes(model, threshold=0.001):
    """Select genes based on feature importance threshold"""
    feature_importances = model.feature_importance.data.cpu().numpy()
    genes_mask = feature_importances >= threshold
    return genes_mask, feature_importances
