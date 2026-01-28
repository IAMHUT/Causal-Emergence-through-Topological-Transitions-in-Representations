# Causal Discovery with Ordering-Based SCM

A PyTorch implementation of causal structure learning using Structural Causal Models (SCM) with constrained ordering weights and variational inference.

## 🎯 Overview

This project implements a neural network-based approach to discover causal relationships from observational data. The model learns both the causal ordering and edge weights simultaneously through a principled optimization framework.

### Key Features

- **Ordering-Based DAG Construction**: Uses learned ordering weights to ensure acyclicity
- **Variational Inference**: Employs reparameterization trick for stable gradient estimation
- **Adaptive Training Strategy**: Dynamic loss weighting across training phases
- **5-Variable Chain Structure**: Demonstrates learning on multi-variable causal chains

## 🏗️ Model Architecture

### Core Components

1. **Encoder Network**: Maps observational data to SCM parameters
   - Input: `n_vars` dimensional observations
   - Output: Ordering weights + Edge weights (with mean and variance)
   
2. **DAG Construction**: 
   - Ordering weights determine causal precedence
   - Edge weights represent causal strengths
   - Sigmoid activation ensures probabilistic interpretation

3. **Loss Function**:
   - DAG fitting loss (MSE between predictions and observations)
   - ELBO loss (KL divergence + negative log-likelihood)
   - Sparsity regularization
   - Ordering consistency constraints

## 📦 Requirements

```bash
torch>=1.9.0
numpy>=1.19.0


from scm_model import SCM, train_model, generate_true_dag_data

# Generate synthetic data (5-variable chain: X1→X2→X3→X4→X5)
X_train, true_scm = generate_true_dag_data(n_samples=800, n_vars=5)

# Initialize model
model = SCM(n_vars=5, hidden_dim=24)

# Train
trained_model = train_model(
    model, 
    X_train, 
    n_epochs=400, 
    lr=0.0001, 
    threshold=0.2,
    batch_size=32
)

# Extract learned causal structure
dag_matrix, order_weights, causal_order = model.get_scm(X_train, threshold=0.2)

