"""Small deterministic unit tests; synthetic rows are only test fixtures."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from src.build_graph import build_graph_windows
from src.data import _binary_labels, load_flows, temporal_window_indices
from src.models.autoencoder import EdgeFeatureAutoencoder, _benign_only_graph, fit_benign_autoencoder
from src.models.gnn import EdgeGraphSAGE, _labeled_mask, fit_supervised
from src.preprocess import FeaturePreprocessor


class PreprocessingTests(unittest.TestCase):
    def test_missing_labels_are_not_mislabeled_as_attacks(self):
        labels = pd.Series(["BENIGN", "DDoS", None, ""])
        np.testing.assert_array_equal(_binary_labels(labels), np.array([0, 1, -1, -1]))

    def test_csv_sampling_ignores_unlabelled_rows(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            df = pd.DataFrame({
                "Timestamp": ["2020-01-01 00:00:00"] * 12,
                "Source IP": ["a"] * 12,
                "Destination IP": ["b"] * 12,
                "Label": ["BENIGN"] * 5 + ["DoS"] * 5 + [None, ""],
                "Flow Duration": list(range(12)),
            })
            df.to_csv(folder / "flows.csv", index=False)
            flows = load_flows(folder, sample=8, seed=3)
            self.assertEqual(len(flows), 8)
            self.assertEqual(flows["_target"].value_counts().to_dict(), {0: 4, 1: 4})
            self.assertTrue(flows["Timestamp"].notna().all())

    def test_identifiers_labels_and_ports_are_not_raw_features(self):
        frame = pd.DataFrame({
            "Source IP": ["10.0.0.1", "10.0.0.2"],
            "Destination IP": ["10.0.0.2", "10.0.0.1"],
            "Source Port": [80, 51000], "Destination Port": [443, 80],
            "Timestamp": ["2020-01-01", "2020-01-02"],
            "Label": ["BENIGN", "DoS"], "Protocol": [6, 6],
            "Flow Bytes/s": [100.0, 1_000_000.0],
        })
        prep = FeaturePreprocessor(engineered=True).fit(frame.iloc[:1])
        x = prep.transform(frame)
        names = prep.output_names
        self.assertFalse(any("Source IP" in name or "Destination IP" in name for name in names))
        self.assertFalse(any("Source Port" in name or "Destination Port" in name for name in names))
        self.assertFalse(any(name in {"Label", "Timestamp"} for name in names))
        self.assertTrue(any("port_bucket" in name for name in names))
        self.assertEqual(x.shape[0], 2)
        self.assertTrue(np.isfinite(x).all())

    def test_fit_must_precede_transform(self):
        prep = FeaturePreprocessor()
        with self.assertRaises(RuntimeError):
            prep.transform(pd.DataFrame({"x": [1]}))


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Source IP": ["a", "b", "a", "c"],
            "Destination IP": ["b", "a", "c", "a"],
            "_target": [0, 1, 0, 1],
        })
        self.features = np.array([[1., 0.], [0., 1.], [1., 1.], [.5, .5]], dtype=np.float32)

    def test_windows_are_directed_flow_labels_with_bidirectional_message_edges(self):
        graphs = build_graph_windows(self.frame, self.features, window_size=2, window_seconds=None)
        self.assertEqual(len(graphs), 2)
        self.assertEqual(graphs[0].edge_index.shape, (2, 4))
        self.assertEqual(graphs[0].edge_x.shape, (2, 2))
        self.assertEqual(graphs[0].y.tolist(), [0, 1])
        self.assertEqual(graphs[1].y.tolist(), [0, 1])
        self.assertEqual(graphs[0].node_x.shape[1], 2)

    def test_timestamp_split_keeps_whole_hour_buckets_together(self):
        base = (1_500_000_000 // 3600) * 3600
        frame = pd.DataFrame({
            "_time_source": ["Timestamp"] * 9,
            "_time_seconds": [base, base + 10, base + 3600, base + 3610, base + 7200, base + 7210, base + 10800, base + 10810, base + 14400],
        })
        tr, va, te = temporal_window_indices(frame, window_seconds=3600, train_fraction=.6, validation_fraction=.2)
        bucket_groups = [set(np.floor(frame.iloc[idx]["_time_seconds"] / 3600).astype(int)) for idx in (tr, va, te)]
        self.assertFalse(bucket_groups[0] & bucket_groups[1])
        self.assertFalse(bucket_groups[0] & bucket_groups[2])
        self.assertFalse(bucket_groups[1] & bucket_groups[2])
        self.assertEqual(sum(map(len, (tr, va, te))), len(frame))

    def test_edge_models_return_one_prediction_or_reconstruction_per_flow(self):
        graph = build_graph_windows(self.frame, self.features, window_size=4, window_seconds=None)[0]
        gnn = EdgeGraphSAGE(input_dim=2, edge_dim=2, hidden=8).eval()
        logits = gnn(graph)
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertTrue(torch.isfinite(logits).all())
        ae = EdgeFeatureAutoencoder(edge_dim=2, hidden=8).eval()
        with torch.no_grad():
            original = ae(graph)
            graph.edge_x = torch.full_like(graph.edge_x, 99.0)
            after_target_change = ae(graph)
        self.assertEqual(tuple(original.shape), (4, 2))
        self.assertTrue(torch.allclose(original, after_target_change))

    def test_autoencoder_training_graph_contains_only_benign_edges(self):
        graph = build_graph_windows(self.frame, self.features, window_size=4, window_seconds=None)[0]
        benign_graph = _benign_only_graph(graph)
        self.assertIsNotNone(benign_graph)
        self.assertEqual(benign_graph.y.tolist(), [0, 0])
        self.assertEqual(benign_graph.edge_index.shape[1], 4)
        self.assertEqual(benign_graph.edge_x.shape[0], 2)

    def test_label_budget_is_class_stratified(self):
        y = torch.tensor([0] * 10 + [1] * 4)
        mask = _labeled_mask(y, .5, np.random.default_rng(7))
        self.assertEqual(int(mask.sum()), 7)
        self.assertEqual(torch.bincount(y[mask], minlength=2).tolist(), [5, 2])

    def test_both_trainers_accept_validation_windows_and_return_best_models(self):
        graph = build_graph_windows(self.frame, self.features, window_size=4, window_seconds=None)[0]
        gnn = fit_supervised([graph], 2, 2, seed=11, epochs=3, hidden=8,
                             validation_graphs=[graph], patience=1)
        ae = fit_benign_autoencoder([graph], 2, 2, seed=11, epochs=3, hidden=8,
                                    validation_graphs=[graph], patience=1)
        self.assertGreaterEqual(gnn.epochs_trained, 1)
        self.assertLessEqual(gnn.epochs_trained, 3)
        self.assertGreaterEqual(ae.epochs_trained, 1)
        self.assertLessEqual(ae.epochs_trained, 3)


if __name__ == "__main__":
    unittest.main()
