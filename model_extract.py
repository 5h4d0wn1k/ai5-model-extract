#!/usr/bin/env python3
"""
AI5 — Model Extraction Tool
Demonstrates query-based model stealing, decision boundary mapping,
architecture inference, and parameter estimation using only numpy.
"""

import numpy as np


class TargetModel:
    """Simulated target model (black-box victim)."""

    def __init__(self, input_size=10, num_classes=4, seed=42):
        rng = np.random.RandomState(seed)
        self.layer_sizes = [input_size, 32, 16, num_classes]
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            w = rng.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.1
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict_proba(self, X):
        current = X
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = self.softmax(z)
        return current

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def query(self, X):
        return self.predict(X)

    def query_confidence(self, X):
        probs = self.predict_proba(X)
        return np.max(probs, axis=1)


class QueryBudget:
    """Tracks query budget usage."""

    def __init__(self, max_queries=10000):
        self.max_queries = max_queries
        self.used = 0

    def can_query(self, n=1):
        return self.used + n <= self.max_queries

    def consume(self, n=1):
        self.used += n

    def remaining(self):
        return self.max_queries - self.used

    def usage_ratio(self):
        return self.used / self.max_queries


class ArchitectureInferrer:
    """Infers model architecture from query behavior."""

    def __init__(self):
        self.inferred_layers = None
        self.activation_guess = None

    def infer_num_classes(self, X_sample):
        unique_outputs = set()
        batch_size = min(50, len(X_sample))
        for i in range(0, len(X_sample), batch_size):
            batch = X_sample[i:i + batch_size]
            preds = batch if isinstance(batch, np.ndarray) else batch
            if hasattr(preds, 'shape'):
                unique_outputs.update(preds.flatten().tolist())
            else:
                unique_outputs.add(preds)
        return len(unique_outputs)

    def infer_by_probing(self, target, n_features, n_classes, budget):
        X_probe = np.random.randn(200, n_features) * 2
        if not budget.can_query(200):
            return {}
        preds = target.query(X_probe)
        budget.consume(200)

        unique_classes = np.unique(preds)
        class_dist = np.bincount(preds, minlength=n_classes) / len(preds)

        confidence = target.query_confidence(X_probe)
        budget.consume(200)

        low_conf = np.mean(confidence < 0.5)
        high_conf = np.mean(confidence > 0.9)

        X_scaled = X_probe * 0.01
        preds_scaled = target.query(X_scaled)
        budget.consume(200)

        X_large = X_probe * 10
        preds_large = target.query(X_large)
        budget.consume(200)

        behavior_consistent = np.mean(preds_scaled == preds_large)

        return {
            "n_classes_detected": len(unique_classes),
            "class_distribution": class_dist.tolist(),
            "mean_confidence": float(np.mean(confidence)),
            "low_confidence_ratio": float(low_conf),
            "high_confidence_ratio": float(high_conf),
            "scale_sensitivity": float(1.0 - behavior_consistent),
            "estimated_depth": "deep" if high_conf > 0.6 else "shallow",
        }

    def infer_nonlinearity(self, target, n_features, budget):
        X = np.random.randn(100, n_features)
        if not budget.can_query(100):
            return "unknown"
        preds1 = target.query(X)
        budget.consume(100)

        X2 = X * 2
        if not budget.can_query(100):
            return "unknown"
        preds2 = target.query(X2)
        budget.consume(100)

        linear_ratio = np.mean(preds1 == preds2)

        X3 = np.clip(X, 0, None)
        if not budget.can_query(100):
            return "unknown"
        preds3 = target.query(X3)
        budget.consume(100)

        relu_ratio = np.mean(preds1 == preds3)

        if linear_ratio > 0.8:
            return "linear/none"
        elif relu_ratio > linear_ratio:
            return "relu"
        else:
            return "non_relu"


class ModelExtractor:
    """Extracts model knowledge through queries."""

    def __init__(self, target, n_features, n_classes, budget):
        self.target = target
        self.n_features = n_features
        self.n_classes = n_classes
        self.budget = budget
        self.query_log_X = []
        self.query_log_y = []

    def random_sampling(self, n_samples=500):
        if not self.budget.can_query(n_samples):
            n_samples = max(0, self.budget.remaining())
        if n_samples <= 0:
            return np.array([]), np.array([])
        X = np.random.randn(n_samples, self.n_features)
        y = self.target.query(X)
        self.budget.consume(n_samples)
        self.query_log_X.append(X)
        self.query_log_y.append(y)
        return X, y

    def adaptive_sampling(self, n_rounds=10, samples_per_round=50):
        if not self.budget.can_query(samples_per_round):
            return np.array([]), np.array([])
        X_all = np.random.randn(samples_per_round, self.n_features)
        y_all = self.target.query(X_all)
        self.budget.consume(samples_per_round)

        for _ in range(n_rounds):
            if not self.budget.can_query(samples_per_round):
                break

            boundary_indices = self._find_boundary_candidates(X_all, y_all)
            if len(boundary_indices) == 0:
                boundary_indices = np.random.choice(
                    len(X_all), size=min(10, len(X_all)), replace=False
                )

            X_new = []
            for idx in boundary_indices:
                perturbation = np.random.randn(self.n_features) * 0.3
                X_new.append(X_all[idx] + perturbation)
                X_new.append(X_all[idx] - perturbation)

            X_new = np.array(X_new)
            n_query = min(len(X_new), self.budget.remaining())
            X_new = X_new[:n_query]
            y_new = self.target.query(X_new)
            self.budget.consume(n_query)

            X_all = np.vstack([X_all, X_new])
            y_all = np.concatenate([y_all, y_new])

        self.query_log_X.append(X_all)
        self.query_log_y.append(y_all)
        return X_all, y_all

    def _find_boundary_candidates(self, X, y, k=10):
        if len(X) < k:
            return list(range(len(X)))

        from collections import Counter
        majority_class = Counter(y.tolist()).most_common(1)[0][0]
        minority_mask = y != majority_class
        if np.sum(minority_mask) == 0:
            return np.random.choice(len(X), size=min(k, len(X)), replace=False)

        minority_indices = np.where(minority_mask)[0]
        majority_indices = np.where(~minority_mask)[0]

        candidates = []
        for midx in minority_indices[:k]:
            dists = np.linalg.norm(X[majority_indices] - X[midx], axis=1)
            nearest = majority_indices[np.argmin(dists)]
            candidates.append(midx)
            candidates.append(nearest)

        return list(set(candidates))

    def boundary_mapping(self, n_directions=50):
        X_boundary = []
        y_boundary = []

        for _ in range(n_directions):
            if not self.budget.can_query(2):
                break
            direction = np.random.randn(self.n_features)
            direction /= np.linalg.norm(direction)
            base = np.random.randn(self.n_features)

            low, high = -5.0, 5.0
            for _ in range(20):
                mid = (low + high) / 2
                point = base + mid * direction
                X_q = point.reshape(1, -1)
                pred = self.target.query(X_q)[0]
                self.budget.consume(1)

                X_q2 = (base + low * direction).reshape(1, -1)
                pred_low = self.target.query(X_q2)[0]
                self.budget.consume(1)

                if pred != pred_low:
                    high = mid
                else:
                    low = mid

            X_boundary.append(base + ((low + high) / 2) * direction)
            X_boundary.append(base + low * direction)
            y_boundary.append(1)
            y_boundary.append(0)

        if len(X_boundary) == 0:
            return np.array([]), np.array([])

        return np.array(X_boundary), np.array(y_boundary)


class ParameterEstimator:
    """Estimates model parameters from extracted data."""

    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes

    def estimate_linear_params(self, X, y):
        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        weights = np.zeros((n_features, n_classes))
        biases = np.zeros(n_classes)

        for c in range(n_classes):
            mask = y == c
            if np.sum(mask) == 0:
                continue
            X_c = X[mask]
            X_other = X[~mask]

            mean_c = np.mean(X_c, axis=0)
            mean_other = np.mean(X_other, axis=0) if np.sum(~mask) > 0 else np.zeros(n_features)
            weights[:, c] = mean_c - mean_other
            biases[c] = -0.5 * (np.dot(mean_c, mean_c) - np.dot(mean_other, mean_other))

        weight_norm = np.linalg.norm(weights)
        if weight_norm > 0:
            weights = weights / weight_norm * 0.5

        return weights, biases

    def estimate_via_regression(self, X, y):
        Y = np.zeros((len(y), self.num_classes))
        Y[np.arange(len(y)), y] = 1
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        pinv = np.linalg.pinv(X_aug)
        params = pinv @ Y
        W = params[:-1, :]
        b = params[-1:, :]
        return W, b.flatten()

    def compute_confidence(self, X, y_true, weights, biases):
        logits = X @ weights + biases
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        accuracy = np.mean(preds == y_true)
        confidence = np.mean(np.max(probs, axis=1))
        return accuracy, confidence


def one_hot(labels, num_classes):
    Y = np.zeros((len(labels), num_classes))
    Y[np.arange(len(labels)), labels] = 1
    return Y


def pseudo_inverse_solve(X, Y):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    pinv = np.linalg.pinv(X_aug)
    params = pinv @ Y
    W = params[:-1, :]
    b = params[-1:, :]
    return W, b.flatten()


class SurrogateModel:
    """Surrogate model trained on extracted knowledge."""

    def __init__(self, input_size, num_classes, hidden_sizes=None):
        if hidden_sizes is None:
            hidden_sizes = [32, 16]
        self.layer_sizes = [input_size] + hidden_sizes + [num_classes]
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.1
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        self.lr = 0.01

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        current = X
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = self.softmax(z)
            self.activations.append(current)
        return current

    def train(self, X, y, epochs=100, batch_size=32):
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_s = X[indices]
            y_s = y[indices]
            for start in range(0, X.shape[0], batch_size):
                end = min(start + batch_size, X.shape[0])
                X_b = X_s[start:end]
                y_b = y_s[start:end]

                probs = self.forward(X_b)
                m = X_b.shape[0]
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(m), y_b] = 1

                delta = probs - one_hot
                grads_w = [None] * len(self.weights)
                grads_b = [None] * len(self.biases)

                grads_w[-1] = self.activations[-2].T @ delta / m
                grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

                for i in range(len(self.weights) - 2, -1, -1):
                    delta = (delta @ self.weights[i + 1].T) * (self.z_values[i] > 0).astype(float)
                    grads_w[i] = self.activations[i].T @ delta / m
                    grads_b[i] = np.sum(delta, axis=0, keepdims=True) / m

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * grads_w[i]
                    self.biases[i] -= self.lr * grads_b[i]

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


def fidelity_score(target_model, surrogate_model, X_test):
    y_target = target_model.query(X_test)
    y_surrogate = surrogate_model.predict(X_test)
    return np.mean(y_target == y_surrogate)


def run_experiment(n_features: int = 10, n_classes: int = 4,
                   max_queries: int = 5000, seed: int = 42,
                   x_test_size: int = 200) -> dict:
    """Run the full model extraction experiment, returning structured results."""
    np.random.seed(seed)
    budget = QueryBudget(max_queries=max_queries)

    target = TargetModel(input_size=n_features, num_classes=n_classes, seed=seed)
    X_test = np.random.randn(x_test_size, n_features)
    y_test = target.query(X_test)

    inferrer = ArchitectureInferrer()
    arch_info = inferrer.infer_by_probing(target, n_features, n_classes, budget)
    activation = inferrer.infer_nonlinearity(target, n_features, budget)

    extractor = ModelExtractor(target, n_features, n_classes, budget)
    n_rand = max(0, min(500, budget.remaining()))
    X_rand, y_rand = extractor.random_sampling(n_samples=n_rand)

    if len(X_rand) == 0:
        fid_random = 0.0
    else:
        surr_random = SurrogateModel(n_features, n_classes)
        surr_random.train(X_rand, y_rand, epochs=80)
        fid_random = float(fidelity_score(target, surr_random, X_test))

    X_adapt, y_adapt = extractor.adaptive_sampling(n_rounds=8, samples_per_round=50)

    if len(X_adapt) == 0:
        fid_adapt = 0.0
    else:
        surr_adapt = SurrogateModel(n_features, n_classes)
        surr_adapt.train(X_adapt, y_adapt, epochs=80)
        fid_adapt = float(fidelity_score(target, surr_adapt, X_test))

    X_boundary, y_boundary = extractor.boundary_mapping(n_directions=30)

    estimator = ParameterEstimator(n_features, n_classes)
    if len(X_rand) == 0:
        W_est = np.zeros((n_features, n_classes))
        b_est = np.zeros(n_classes)
        acc_est = 0.0
        conf_est = 0.0
    else:
        W_est, b_est = estimator.estimate_linear_params(X_rand, y_rand)
        acc_est, conf_est = estimator.compute_confidence(X_test, y_test, W_est, b_est)

    return {
        "target": {
            "n_features": n_features,
            "n_classes": n_classes,
            "max_queries": max_queries,
            "seed": seed,
        },
        "architecture_inference": {
            **{k: v for k, v in arch_info.items()},
            "nonlinearity": activation,
        },
        "extraction": {
            "random_samples": int(len(X_rand)),
            "adaptive_samples_total": int(len(X_adapt)),
            "boundary_points_mapped": int(len(X_boundary)),
        },
        "accuracy_transfer": {
            "fidelity_random_sampling": fid_random,
            "fidelity_adaptive_sampling": fid_adapt,
            "improvement": fid_adapt - fid_random,
        },
        "parameter_estimation": {
            "linear_accuracy": float(acc_est),
            "linear_confidence": float(conf_est),
            "weight_matrix_shape": list(W_est.shape),
            "bias_shape": list(b_est.shape),
        },
        "query_budget": {
            "max_queries": budget.max_queries,
            "used": budget.used,
            "remaining": budget.remaining(),
            "usage_ratio": float(budget.usage_ratio()),
        },
    }


def format_report(results: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("AI5 — Model Extraction Tool Demonstration")
    lines.append("=" * 60)
    t = results["target"]
    lines.append(f"\n[1] Architecture inference: {t['n_features']} features, "
                 f"{t['n_classes']} classes")
    ai = results["architecture_inference"]
    lines.append(f"    Classes detected: {ai['n_classes_detected']}")
    lines.append(f"    Mean confidence:  {ai['mean_confidence']:.4f}")
    lines.append(f"    Scale sensitivity: {ai['scale_sensitivity']:.4f}")
    lines.append(f"    Estimated depth:  {ai['estimated_depth']}")
    lines.append(f"    Nonlinearity:     {ai['nonlinearity']}")

    lines.append(f"\n[2] Random sampling extraction "
                 f"(budget: {results['query_budget']['remaining']})...")
    lines.append(f"    Collected {results['extraction']['random_samples']} samples")
    lines.append(f"    Fidelity (random): "
                 f"{results['accuracy_transfer']['fidelity_random_sampling']:.4f}")

    lines.append(f"\n[3] Adaptive sampling extraction...")
    lines.append(f"    Collected {results['extraction']['adaptive_samples_total']} samples")
    lines.append(f"    Fidelity (adaptive): "
                 f"{results['accuracy_transfer']['fidelity_adaptive_sampling']:.4f}")

    lines.append(f"\n[4] Decision boundary mapping...")
    lines.append(f"    Boundary points mapped: {results['extraction']['boundary_points_mapped']}")

    lines.append(f"\n[5] Parameter estimation (linear fit)...")
    pe = results["parameter_estimation"]
    lines.append(f"    Linear model accuracy: {pe['linear_accuracy']:.4f}")
    lines.append(f"    Linear model confidence: {pe['linear_confidence']:.4f}")

    lines.append(f"\n[6] Query budget summary...")
    qb = results["query_budget"]
    lines.append(f"    Used queries:    {qb['used']}")
    lines.append(f"    Remaining:       {qb['remaining']}")
    lines.append(f"    Usage ratio:     {qb['usage_ratio']:.2%}")

    lines.append(f"\n[7] Fidelity comparison...")
    at = results["accuracy_transfer"]
    lines.append(f"    Random sampling:  {at['fidelity_random_sampling']:.4f}")
    lines.append(f"    Adaptive sampling: {at['fidelity_adaptive_sampling']:.4f}")
    lines.append(f"    Improvement:      {at['improvement']:+.4f}")

    lines.append("\n" + "=" * 60)
    lines.append("Demonstration complete.")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        prog="ai5-model-extract",
        description="Query-based model extraction research: surrogate training, "
                    "fidelity/accuracy transfer, architecture inference. "
                    "Offline, self-contained.")
    parser.add_argument("--features", type=int, default=10,
                        help="input feature count")
    parser.add_argument("--classes", type=int, default=4,
                        help="target class count")
    parser.add_argument("--queries", type=int, default=5000,
                        help="query budget")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", metavar="FILE",
                        help="write JSON report to FILE (e.g. reports/ai5-report.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress human-readable output")
    args = parser.parse_args(argv)

    results = run_experiment(n_features=args.features, n_classes=args.classes,
                             max_queries=args.queries, seed=args.seed)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    if not args.quiet:
        print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
