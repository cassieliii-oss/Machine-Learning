"""
=============================================================================
Machine Learning Homework: Linear Regression & Generalization
=============================================================================

In this assignment, you will:
1. Explore a regression dataset with 231 features
2. Train linear models using SGD (Stochastic Gradient Descent)
3. Experiment with feature selection and regularization
4. Experience the train/validation/test gap firsthand
5. Submit your best weight vector to the class leaderboard

You are given:
  - train.csv                  : 500 samples, 231 features + target (y)

The evaluation server holds two data sets:
  - Client sample:  your MSE is shown on the leaderboard after each submission
  - Live data:      revealed only after the deadline (simulates deployment)

Your model: y = X @ w   (a linear model, 231 weights)
Your goal:  minimize MSE on the live data — but you won't see it
            until the deadline, so build a model that generalizes.

RULES:
  - You may only use linear models: y = X @ w
  - You must implement or use SGD for training.
  - You may use any feature selection, regularization, or validation
    strategy you want.

HINT: Not all 231 features are useful. Some might hurt your model.
=============================================================================
"""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


def load_data(path: str):
    train_df = pd.read_csv(path)
    X = train_df.drop(columns=["y"]).to_numpy(dtype=float)
    y = train_df["y"].to_numpy(dtype=float)
    feature_names = [c for c in train_df.columns if c != "y"]
    return X, y, feature_names


# =============================================================================
# 2. Standardization Helpers
# =============================================================================

def find_bias_col(X):
    """Find the constant (bias) column in a feature matrix."""
    return int(np.argmin(np.std(X, axis=0)))


def standardize(X_train, X_other=None, *, bias_col: int):
    """
    Standardize features to zero-mean, unit-variance.

    IMPORTANT: The constant (bias) column is left untouched so the model
    can learn an intercept. If you standardize it, it becomes all zeros
    and the model loses its ability to fit an intercept — this silently
    adds a huge error equal to (mean of y)^2.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    mu[bias_col] = 0.0
    sigma[bias_col] = 1.0

    X_train_norm = (X_train - mu) / sigma
    if X_other is not None:
        return X_train_norm, (X_other - mu) / sigma, mu, sigma
    return X_train_norm, mu, sigma


def to_raw_weights(w_norm, mu, sigma, *, bias_col: int):
    """
    Convert weights from standardized space back to raw feature space.

    The server computes y = X_raw @ w (all 231 features), so we need:
    1. Undo the standardization: w_raw[j] = w_norm[j] / sigma[j]
    2. Absorb the centering into the bias weight:
       w_bias = w_bias_norm - sum(mu_j * w_norm_j / sigma_j)
    """
    w_raw = w_norm / sigma
    w_raw[bias_col] = w_norm[bias_col] - np.sum(mu * w_norm / sigma)
    return w_raw


# =============================================================================
# 3. SGD Implementation
# =============================================================================

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def select_topk_by_corr(X_raw, y, *, top_k: int | None, bias_col: int):
    """
    Simple feature selection: keep the top-k features by |corr(x_j, y)|.
    We still output a 231-dim weight vector; "dropped" features just get w_j = 0.
    """
    d = X_raw.shape[1]
    mask = np.ones(d, dtype=bool)
    if top_k is None or top_k >= d:
        return mask

    # Exclude bias from scoring (it is constant, corr is undefined)
    feat_idx = np.array([j for j in range(d) if j != bias_col], dtype=int)
    Xc = X_raw[:, feat_idx]
    yc = y - y.mean()
    Xc_center = Xc - Xc.mean(axis=0)
    denom = (np.linalg.norm(Xc_center, axis=0) * (np.linalg.norm(yc) + 1e-12)) + 1e-12
    corr = (Xc_center.T @ yc) / denom
    keep_feat = feat_idx[np.argsort(np.abs(corr))[-top_k:]]

    mask[:] = False
    mask[bias_col] = True
    mask[keep_feat] = True
    return mask


@dataclass(frozen=True)
class SGDConfig:
    lr: float = 3e-3
    epochs: int = 600
    batch_size: int = 64
    l2_lambda: float = 1e-3
    l1_lambda: float = 0.0
    momentum: float = 0.9
    lr_decay: float = 1e-3
    average: bool = True
    top_k: int | None = None
    grad_clip_norm: float | None = 5.0
    seed: int = 0


def sgd(X, y, *, cfg: SGDConfig, bias_col: int, feature_mask=None):
    """
    Mini-batch SGD for linear regression with optional L2/L1 regularization.

    Optimizations explored (still a linear model y = X @ w trained with SGD):
    - Feature standardization (excluding the bias/constant column): stabilizes SGD and makes the learning rate easier to tune.
    - L2 regularization (Ridge): reduces overfitting to noisy features and often improves validation/client performance.
    - L1 regularization (soft-thresholding / Lasso-style): encourages sparsity, acting like implicit feature selection.
    - Learning-rate decay: fast initial progress with more stable late-stage convergence.
    - Polyak weight averaging: reduces SGD noise and can lower validation MSE.
    - Simple top-k correlation feature selection: drops clearly uninformative dimensions to reduce variance.

    The “best” choice is selected by cross-validation (random search + K-fold CV below).
    """
    n, d = X.shape
    rng = np.random.default_rng(cfg.seed)

    w = rng.normal(loc=0.0, scale=0.01, size=d)
    v = np.zeros_like(w)

    # Do not regularize bias
    reg_mask = np.ones(d, dtype=bool)
    reg_mask[bias_col] = False

    if feature_mask is not None:
        feature_mask = feature_mask.astype(bool)
        feature_mask[bias_col] = True
        w[~feature_mask] = 0.0
    else:
        feature_mask = None

    t = 0
    w_avg = np.zeros_like(w)
    avg_count = 0
    burn_in = 50  # start averaging after a few updates

    losses = []
    for epoch in range(cfg.epochs):
        perm = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            Xb = X[idx]
            yb = y[idx]

            pred = Xb @ w
            err = pred - yb
            grad = (2.0 / len(idx)) * (Xb.T @ err)

            if cfg.l2_lambda > 0:
                grad[reg_mask] += 2.0 * cfg.l2_lambda * w[reg_mask]

            if feature_mask is not None:
                grad[~feature_mask] = 0.0

            if cfg.grad_clip_norm is not None:
                gnorm = float(np.linalg.norm(grad))
                if np.isfinite(gnorm) and gnorm > cfg.grad_clip_norm:
                    grad = grad * (cfg.grad_clip_norm / (gnorm + 1e-12))

            lr_t = cfg.lr / (1.0 + cfg.lr_decay * t)

            if cfg.momentum > 0:
                v = cfg.momentum * v + grad
                w = w - lr_t * v
            else:
                w = w - lr_t * grad

            if cfg.l1_lambda > 0:
                threshold = lr_t * cfg.l1_lambda
                w[reg_mask] = np.sign(w[reg_mask]) * np.maximum(np.abs(w[reg_mask]) - threshold, 0.0)

            if feature_mask is not None:
                w[~feature_mask] = 0.0

            if cfg.average and t >= burn_in:
                avg_count += 1
                w_avg += (w - w_avg) / avg_count

            t += 1

        if not np.isfinite(w).all():
            # Diverged; return a clearly-bad solution for CV
            return np.zeros(d), np.array([float("inf")], dtype=float)

        losses.append(mse(y, X @ w))

    w_out = w_avg if (cfg.average and avg_count > 0) else w
    return w_out, np.array(losses, dtype=float)


# =============================================================================
# 4. Evaluation Helpers
# =============================================================================

def train_val_split(X, y, val_fraction=0.2, seed=42):
    """Simple random train/val split."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    n_val = int(len(y) * val_fraction)
    return X[perm[n_val:]], y[perm[n_val:]], X[perm[:n_val]], y[perm[:n_val]]


def k_fold_cv(X, y, k=5, seed=42, **sgd_kwargs):
    """
    K-fold cross-validation. Returns (mean_mse, std_mse) across folds.
    Handles standardization properly per fold.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(y))
    fold_size = len(y) // k
    val_scores = []

    for i in range(k):
        val_idx = indices[i * fold_size : (i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        bias_col = sgd_kwargs["bias_col"]
        top_k = sgd_kwargs["cfg"].top_k
        feat_mask = select_topk_by_corr(X_tr, y_tr, top_k=top_k, bias_col=bias_col)

        X_tr_n, X_va_n, _, _ = standardize(X_tr, X_va, bias_col=bias_col)
        w, _ = sgd(X_tr_n, y_tr, cfg=sgd_kwargs["cfg"], bias_col=bias_col, feature_mask=feat_mask)
        val_scores.append(mse(y_va, X_va_n @ w))

    return np.mean(val_scores), np.std(val_scores)




def random_search_configs(*, rng: np.random.Generator, trials: int):
    # A compact search space that tends to work well on this dataset size/scale.
    # NOTE: Larger learning rates (e.g. >1e-2) can easily diverge even after standardization.
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    epochs = [400, 600, 800]
    batch_sizes = [16, 32, 64, 128]
    l2s = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    l1s = [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    momenta = [0.0, 0.9]
    decays = [0.0, 1e-4, 3e-4, 1e-3]
    topks = [None, 40, 60, 80, 100, 120, 160]
    averages = [True]

    for _ in range(trials):
        mom = float(rng.choice(momenta))
        lr_choices = lrs if mom == 0.0 else [1e-4, 3e-4, 1e-3, 3e-3]
        top_k = rng.choice(topks)
        yield SGDConfig(
            lr=float(rng.choice(lr_choices)),
            epochs=int(rng.choice(epochs)),
            batch_size=int(rng.choice(batch_sizes)),
            l2_lambda=float(rng.choice(l2s)),
            l1_lambda=float(rng.choice(l1s)),
            momentum=mom,
            lr_decay=float(rng.choice(decays)),
            average=bool(rng.choice(averages)),
            top_k=None if top_k is None else int(top_k),
            seed=int(rng.integers(0, 10_000)),
        )


def choose_best_config(X, y, *, bias_col: int, k: int, cv_repeats: int, trials: int, seed: int):
    rng = np.random.default_rng(seed)

    best = None
    best_mean = float("inf")

    tried = 0
    for cfg in random_search_configs(rng=rng, trials=trials):
        scores = []
        for r in range(cv_repeats):
            mean_mse, _ = k_fold_cv(X, y, k=k, seed=seed + r, cfg=cfg, bias_col=bias_col)
            scores.append(mean_mse)
        mean = float(np.mean(scores))
        tried += 1

        if mean < best_mean:
            best_mean = mean
            best = cfg
            print(f"[CV best @ trial {tried:03d}] mean_mse={best_mean:.4f} cfg={best}")

    return best, best_mean


def main():
    parser = argparse.ArgumentParser(description="Linear Regression via SGD (Assignment 1)")
    parser.add_argument("--data", default="train.csv", help="Path to train.csv")
    parser.add_argument("--name", default="cassieli", help='Leaderboard name prefix, output will be "<name>_IP04.npy"')
    parser.add_argument("--out", default=None, help="Output .npy path (overrides --name)")
    parser.add_argument("--k", type=int, default=5, help="K for K-fold CV")
    parser.add_argument("--cv_repeats", type=int, default=2, help="Repeat CV with different shuffles")
    parser.add_argument("--trials", type=int, default=60, help="Random hyperparameter trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for search")
    args = parser.parse_args()

    # 1) Load
    X, y, feature_names = load_data(args.data)
    n_samples, n_features = X.shape
    bias_col = find_bias_col(X)

    print(f"Training data: {n_samples} samples, {n_features} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}, std={y.std():.2f}")
    print(f"Constant column (bias): {feature_names[bias_col]} (index={bias_col}, value={X[0, bias_col]})")

    # 2) Hyperparameter search via CV
    print("\n" + "=" * 72)
    print("Hyperparameter search (random search + K-fold CV)")
    print("=" * 72)
    best_cfg, best_cv = choose_best_config(
        X, y, bias_col=bias_col, k=args.k, cv_repeats=args.cv_repeats, trials=args.trials, seed=args.seed
    )

    assert best_cfg is not None
    print("\n" + "=" * 72)
    print(f"Best config by CV: mean_mse={best_cv:.4f}")
    print(best_cfg)
    print("=" * 72)

    # =========================================================================
    # Optimization notes for the Canvas submission (in-code write-up)
    # =========================================================================
    # Optimizations tried & why (all maintain the linear form y = X @ w and use SGD updates):
    # - Standardization (excluding the bias/constant column): keeps gradient scales comparable across features; improves SGD stability.
    # - L2 regularization (Ridge): penalizes large weights to reduce variance / overfitting from noisy features.
    # - L1 regularization (soft-thresholding): encourages sparse solutions, effectively dropping weak features.
    # - Learning-rate decay: reduces oscillation and improves late-stage convergence.
    # - Polyak averaging: averages iterates to reduce stochastic noise and often improves validation MSE.
    # - Top-k correlation feature selection: removes clearly uninformative dimensions, reducing variance and improving generalization.
    # - Gradient clipping + divergence checks: prevents numerical blow-ups (some larger learning rates can overflow).
    #
    # Best result selection criterion: lowest mean MSE from K-fold cross-validation (best_cfg / best_cv).
    # In the best-performing runs, the most impactful pieces were typically:
    # - Mild regularization (often small L2; L1 either 0 or very small) plus a smaller top-k (e.g., 40) to reduce overfitting.
    # - Polyak averaging + learning-rate decay for a more stable optimizer and better robustness to noise.
    # Why this helps: the dataset has high dimensionality (231 features) but only 500 samples, so many features behave like noise.
    #              Regularization/feature selection reduces variance, and averaging/decay reduces optimization noise => better validation/client MSE.

    # 3) Final train on full data using best config
    feat_mask = select_topk_by_corr(X, y, top_k=best_cfg.top_k, bias_col=bias_col)
    X_n, mu, sigma = standardize(X, bias_col=bias_col)
    w_final, losses = sgd(X_n, y, cfg=best_cfg, bias_col=bias_col, feature_mask=feat_mask)

    train_mse = mse(y, X_n @ w_final)
    print(f"\nTrain MSE (on standardized X): {train_mse:.4f}")

    # Convert to raw feature space for submission
    w_raw = to_raw_weights(w_final, mu, sigma, bias_col=bias_col)
    raw_mse = mse(y, X @ w_raw)
    print(f"Sanity check (raw-space MSE, should match above): {raw_mse:.4f}")

    # 4) Save for submission
    out_path = args.out or f"{args.name}_IP04.npy"
    np.save(out_path, w_raw.astype(np.float64))
    print(f"\nSaved submission weights to: {out_path}  shape={w_raw.shape}")

    # One-line reminder for the leaderboard
    print("Upload this .npy file to the leaderboard server.")


if __name__ == "__main__":
    main()
