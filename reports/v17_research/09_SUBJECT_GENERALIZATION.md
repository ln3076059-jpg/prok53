# 09. Subject Generalization via MixStyle and Group-DRO

## 1. The Subject Collapse Problem
In baseline V16, Subject `gZ-36` exhibited a catastrophic SAFE_DRIVING recall of **1.66%**, while pooled recall was **60.38%**. Standard empirical risk minimization (ERM) ignores such subgroup failures.

## 2. Group-DRO (Distributionally Robust Optimization)
Instead of minimizing average loss $\frac{1}{N} \sum_i L_i$, Group-DRO minimizes the maximum loss across driver groups:
$$\min_\theta \max_{g \in \mathcal{G}} \mathbb{E}_{(x, y) \sim P_g} [L(f_\theta(x), y)]$$
This mathematically forces the model to learn features that work for the most difficult driver rather than over-optimizing for the majority.

## 3. MixStyle Feature Regularization
MixStyle interpolates feature statistics between two distinct drivers in intermediate layers:
$$\mu_{\text{mix}} = \lambda \mu(x_i) + (1-\lambda) \mu(x_j), \quad \sigma_{\text{mix}} = \lambda \sigma(x_i) + (1-\lambda) \sigma(x_j)$$
Synthesizing virtual cabin appearances and clothing textures during training.
