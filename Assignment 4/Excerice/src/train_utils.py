import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from tqdm import tqdm


def save_metrics_to_file(metrics, filename):
    """Save the metrics dictionary to a JSON file."""
    # Initialize an empty dictionary to hold metrics
    all_metrics = {}

    # Check if the file exists
    if os.path.exists(filename):
        # Load existing metrics
        with open(filename, 'r') as f:
            all_metrics = json.load(f)

    # Append or update the new metrics
    for key, values in metrics.items():
        all_metrics[key] = values  # Add new key-value pair

    # Save updated metrics back to the file
    with open(filename, 'w') as f:
        json.dump(all_metrics, f, indent=4)

def compute_covariance(features: []):
    ######################
    # Implement Coral loss
    ######################
    assert len(features) == 2
    source_features, target_features = features

    # Number of samples in source and target
    n_S = source_features.size(0)
    n_T = target_features.size(0)

    # Compute the covariance matrix for the source domain
    source_mean = torch.mean(source_features, dim=0, keepdim=True)  # [1, d]
    source_centered = source_features - source_mean
    source_cov = (source_centered.T @ source_centered) / (n_S - 1)

    # Compute the covariance matrix for the target domain
    target_mean = torch.mean(target_features, dim=0, keepdim=True)  # [1, d]
    target_centered = target_features - target_mean
    target_cov = (target_centered.T @ target_centered) / (n_T - 1)

    # Calculate CORAL loss as the squared Frobenius norm of the difference
    d = source_features.size(1)
    loss = torch.sum((source_cov - target_cov) ** 2) / (4 * d * d)
    return loss

def train_baseline(model, source_loader, target_loader, args, device):
    """Standard source training"""
    print("\nTraining Baseline Model...")

    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    metrics = defaultdict(list)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for data, target in tqdm(source_loader, desc=f'Epoch {epoch}'):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate training and testing metrics
        source_loss, source_acc = evaluate(model, source_loader, device)
        target_loss, target_acc = evaluate(model, target_loader, device)

        # Print loss and accuracy for source and target 
        print(f"Epoch {epoch} - Source Loss: {source_loss:.4f}, Source Acc: {source_acc:.4f}, Target Loss: {target_loss:.4f}, Target Acc: {target_acc:.4f}")

        # Metrics for report
        metrics['source_loss'].append(source_loss)
        metrics['source_acc'].append(source_acc)
        metrics['target_loss'].append(target_loss)
        metrics['target_acc'].append(target_acc)
        metrics['total_loss'].append(total_loss)

    save_metrics_to_file({'baseline': metrics}, 'metrics.json')

    # Save final model
    torch.save(model.state_dict(), 'final_baseline.pth')

    # Return Final target accuracy 
    return target_acc

def train_coral(model, source_loader, target_loader, args, device):
    """CORAL training"""
    print("\nTraining CORAL Model...")

    metrics = defaultdict(list)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_coral_loss = 0

        for (source_data, source_target), (target_data, _) in zip(source_loader, target_loader):

            source_data = source_data.to(device)
            source_target = source_target.to(device)
            target_data = target_data.to(device)

            optimizer.zero_grad()

            # Extract features
            source_features = model.feature_extractor(source_data)
            target_features = model.feature_extractor(target_data)

            # Classification loss
            source_outputs = model.classifier(source_features)
            cls_loss = F.nll_loss(source_outputs, source_target)

            # CORAL loss
            # I had to add to the template, because the method you proposed didn't allow for the loss to be computed
            # Loss taken from section 4 of https://arxiv.org/pdf/1612.01939
            coral_loss = compute_covariance([source_features, target_features])

            # Total loss
            loss = cls_loss + args.coral_weight * coral_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_coral_loss += coral_loss.item()

        # Calculate training and testing metrics
        source_loss, source_acc = evaluate(model, source_loader, device)
        target_loss, target_acc = evaluate(model, target_loader, device)

        # Print loss and accuracy for source and target
        print(f"Epoch {epoch} - Source Loss: {source_loss:.4f}, Source Acc: {source_acc:.4f}, Target Loss: {target_loss:.4f}, Target Acc: {target_acc:.4f}")

        # Metrics for report
        metrics['source_loss'].append(source_loss)
        metrics['source_acc'].append(source_acc)
        metrics['target_loss'].append(target_loss)
        metrics['target_acc'].append(target_acc)
        metrics['total_loss'].append(total_loss)
        metrics['total_cls_loss'].append(total_cls_loss)
        metrics['total_coral_loss'].append(total_coral_loss)

    save_metrics_to_file({'coral': metrics}, 'metrics.json')

    # Save final model
    torch.save(model.state_dict(), 'final_coral.pth')

    # Return Final target accuracy 
    return target_acc

def train_adversarial(model, source_loader, target_loader, args, device):
    """Adversarial training"""
    print("\nTraining Adversarial Model...")

    discriminator = nn.Sequential(
        nn.Linear(256, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 2),
    ).to(device)

    optimizer_g = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)

    metrics = defaultdict(list)
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        total_loss_g = 0
        total_loss_d = 0
        total_cls_loss = 0
        total_gen_loss = 0

        for (source_data, source_target), (target_data, _) in zip(source_loader, target_loader):
            source_data = source_data.to(device)
            source_target = source_target.to(device)
            target_data = target_data.to(device)
            batch_size = source_data.size(0)

            # Train discriminator
            optimizer_d.zero_grad()

            source_features = model.feature_extractor(source_data).detach()
            target_features = model.feature_extractor(target_data).detach()

            source_domain = torch.zeros(batch_size).long().to(device)
            target_domain = torch.ones(batch_size).long().to(device)

            source_domain_pred = discriminator(source_features)
            target_domain_pred = discriminator(target_features)

            d_loss = F.cross_entropy(source_domain_pred, source_domain) + F.cross_entropy(target_domain_pred, target_domain)

            d_loss.backward()
            optimizer_d.step()
            total_loss_d += d_loss.item()

            # Train generator
            optimizer_g.zero_grad()

            source_features = model.feature_extractor(source_data)
            target_features = model.feature_extractor(target_data)
            source_outputs = model.classifier(source_features)

            # Classification loss
            cls_loss = F.nll_loss(source_outputs, source_target)

            # Adversarial loss
            source_domain_pred = discriminator(source_features)
            target_domain_pred = discriminator(target_features)

            gen_loss = F.cross_entropy(source_domain_pred, target_domain) + F.cross_entropy(target_domain_pred, source_domain)

            loss_g = cls_loss + args.adversarial_weight * gen_loss
            loss_g.backward()
            optimizer_g.step()
            total_loss_g += loss_g.item()
            total_cls_loss += cls_loss.item()
            total_gen_loss += gen_loss.item()

        # Calculate training and testing metrics
        source_loss, source_acc = evaluate(model, source_loader, device)
        target_loss, target_acc = evaluate(model, target_loader, device)

        # Print loss and accuracy for source and target
        print(f"Epoch {epoch} - Source Loss: {source_loss:.4f}, Source Acc: {source_acc:.4f}, Target Loss: {target_loss:.4f}, Target Acc: {target_acc:.4f}")

        # Metrics for report
        metrics['source_loss'].append(source_loss)
        metrics['source_acc'].append(source_acc)
        metrics['target_loss'].append(target_loss)
        metrics['target_acc'].append(target_acc)
        metrics['total_loss_d'].append(total_loss_d)
        metrics['total_loss_g'].append(total_loss_g)
        metrics['total_cls_loss_g'].append(total_cls_loss)
        metrics['total_gen_loss_g'].append(total_gen_loss)

    save_metrics_to_file({'adverserial': metrics}, 'metrics.json')

    # Save final model
    torch.save({
        'model': model.state_dict(),
        'discriminator': discriminator.state_dict()
    }, 'final_adversarial.pth')

    # Return Final target accuracy 
    return target_acc


def train_adabn(model, source_loader, target_loader, args, device):
    """AdaBN with source training and target adaptation"""
    print("\nTraining AdaBN Model...")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 1. Train on source
    metrics = defaultdict(list)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for data, target in tqdm(source_loader, desc=f'Epoch {epoch} (Source Training)'):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        metrics['total_loss'].append(total_loss)

    # 2. Adapt BN statistics on target domain
    model.train()
    print("\nAdapting BN statistics on target domain...")

    ########################################################
    # Implement AdaBN (forward on target data for args.epoch)
    ######################################################
    with torch.no_grad():
        for _ in range(args.epochs):
            for target_data, _ in target_loader:
                target_data = target_data.to(device)
                model(target_data)  # Forward pass to update BN stats

    # Calculate target accuracy and print it 
    source_loss, source_acc = evaluate(model, source_loader, device)
    target_loss, target_acc = evaluate(model, target_loader, device)
    print(f"Final Target Accuracy after BN Adaptation: {target_acc:.4f}")

    # Metrics for report
    metrics['source_loss'].append(source_loss)
    metrics['source_acc'].append(source_acc)
    metrics['target_loss'].append(target_loss)
    metrics['target_acc'].append(target_acc)

    save_metrics_to_file({'adabn': metrics}, 'metrics.json')

    # Save final model
    torch.save(model.state_dict(), 'final_adabn.pth')

    # Return Final target accuracy 
    return target_acc

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target).item()
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), correct / total