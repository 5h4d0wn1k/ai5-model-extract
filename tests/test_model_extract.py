"""AI5 model-extraction engine tests — real code paths, offline, stdlib only."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_extract import (  # noqa: E402
    ArchitectureInferrer,
    ModelExtractor,
    ParameterEstimator,
    QueryBudget,
    SurrogateModel,
    TargetModel,
    fidelity_score,
    run_experiment,
)


class TestTargetModel(unittest.TestCase):
    def test_query_and_confidence_shapes(self):
        target = TargetModel(input_size=8, num_classes=3, seed=1)
        X = np.random.randn(20, 8)
        preds = target.query(X)
        conf = target.query_confidence(X)
        self.assertEqual(preds.shape, (20,))
        self.assertEqual(conf.shape, (20,))
        self.assertTrue(np.all(conf <= 1.0))


class TestQueryBudget(unittest.TestCase):
    def test_budget_tracks_usage(self):
        budget = QueryBudget(max_queries=10)
        budget.consume(3)
        self.assertEqual(budget.remaining(), 7)
        self.assertEqual(budget.usage_ratio(), 0.3)
        self.assertTrue(budget.can_query(2))
        self.assertFalse(budget.can_query(8))


class TestExtraction(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)
        self.target = TargetModel(input_size=8, num_classes=3, seed=2)
        self.budget = QueryBudget(max_queries=2000)

    def test_architecture_inference_fields(self):
        inferrer = ArchitectureInferrer()
        info = inferrer.infer_by_probing(self.target, 8, 3, self.budget)
        self.assertIn("n_classes_detected", info)
        self.assertIn("mean_confidence", info)
        self.assertIn("estimated_depth", info)

    def test_surrogate_trains_and_predicts(self):
        extractor = ModelExtractor(self.target, 8, 3, self.budget)
        X, y = extractor.random_sampling(n_samples=60)
        surr = SurrogateModel(8, 3)
        surr.train(X, y, epochs=10)
        self.assertEqual(surr.predict(X[:5]).shape, (5,))

    def test_fidelity_score_in_bounds(self):
        extractor = ModelExtractor(self.target, 8, 3, self.budget)
        X, y = extractor.random_sampling(n_samples=60)
        surr = SurrogateModel(8, 3)
        surr.train(X, y, epochs=10)
        X_test = np.random.randn(50, 8)
        score = fidelity_score(self.target, surr, X_test)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestRunExperiment(unittest.TestCase):
    def test_returns_structured_results(self):
        r = run_experiment(n_features=6, n_classes=3, max_queries=1000,
                           seed=1)
        self.assertIn("architecture_inference", r)
        self.assertIn("accuracy_transfer", r)
        self.assertIn("fidelity_random_sampling", r["accuracy_transfer"])
        self.assertIn("parameter_estimation", r)
        self.assertIn("query_budget", r)

    def test_budget_never_exceeded(self):
        r = run_experiment(n_features=6, n_classes=3, max_queries=1000, seed=1)
        self.assertLessEqual(r["query_budget"]["used"],
                             r["query_budget"]["max_queries"])


class TestCLI(unittest.TestCase):
    def test_cli_writes_json_report_and_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.json")
            from model_extract import main
            code = main(["--features", "6", "--classes", "3",
                         "--queries", "800", "--seed", "1",
                         "--output", out, "--quiet"])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["target"]["n_features"], 6)


if __name__ == "__main__":
    unittest.main()