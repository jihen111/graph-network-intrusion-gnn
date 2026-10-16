"""Small deterministic unit tests; synthetic rows are only test fixtures."""
import unittest

import numpy as np
import pandas as pd
import torch

from src.build_graph import N_STRUCT, build_graph_windows, flow_order
from src.data import (_binary_labels, add_windows, normalize_attack_name, repair_cic_timestamps,
                      split_protocol_a, split_protocol_b, window_benign_cap)
from src.metrics import classification_metrics, threshold_at_fpr, threshold_max_f1
from src.models.autoencoder import EdgeFeatureAutoencoder, _benign_only_graph, fit_benign_autoencoder
from src.models.gnn import EdgeGraphSAGE, _labeled_mask, fit_supervised
from src.preprocess import FeaturePreprocessor


def toy_frame():
    return pd.DataFrame({
        "Source IP": ["a", "b", "a", "c", "a", "a"],
        "Destination IP": ["b", "a", "c", "a", "d", "e"],
        "Destination Port": [80, 443, 22, 80, 21, 23],
        "Source Port": [5000, 5001, 5002, 5003, 5004, 5005],
        "_target": [0, 1, 0, 1, 1, 1],
        "_window": [0, 0, 0, 0, 1, 1],
    })


class DataTests(unittest.TestCase):
    def test_missing_labels_are_not_mislabeled_as_attacks(self):
        np.testing.assert_array_equal(_binary_labels(pd.Series(["BENIGN", "DDoS", None, ""])), np.array([0, 1, -1, -1]))

    def test_twelve_hour_clock_is_repaired(self):
        ts = repair_cic_timestamps(pd.Series(["7/7/2017 3:56", "7/7/2017 10:02", "3/7/2017 08:55:58"]))
        self.assertEqual(ts.dt.hour.tolist(), [15, 10, 8])
        self.assertEqual(ts.dt.dayofweek.tolist(), [4, 4, 0])  # Friday, Friday, Monday

    def test_attack_names_are_normalized(self):
        self.assertEqual(normalize_attack_name("Web Attack � Brute Force"), "Web Attack - Brute Force")
        self.assertEqual(normalize_attack_name("BENIGN"), "Benign")

    def test_benign_cap_keeps_every_attack_flow(self):
        frame = pd.DataFrame({"_window": [0] * 10 + [1] * 10, "_target": [0] * 8 + [1] * 2 + [0] * 10,
                              "_time_seconds": np.arange(20)})
        capped = window_benign_cap(frame, 3, seed=1)
        self.assertEqual(int(capped["_target"].sum()), 2)
        self.assertEqual(capped.groupby("_window")["_target"].apply(lambda s: (s == 0).sum()).tolist(), [3, 3])

    def test_protocol_a_uses_days(self):
        frame = pd.DataFrame({"_day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]})
        tr, va, te = split_protocol_a(frame)
        self.assertEqual((tr.tolist(), va.tolist(), te.tolist()), ([0, 1, 2], [3], [4]))

    def test_protocol_b_puts_each_attack_type_in_every_split_without_sharing_windows(self):
        rows = []
        for w in range(20):
            kind = "DoS" if w < 10 else "Benign"
            rows += [{"_window": w, "_target": int(kind != "Benign"), "_attack_type": kind},
                     {"_window": w, "_target": 0, "_attack_type": "Benign"}]
        frame = pd.DataFrame(rows)
        parts = split_protocol_b(frame)
        windows = [set(frame.iloc[p]["_window"]) for p in parts]
        self.assertFalse(windows[0] & windows[1] or windows[0] & windows[2] or windows[1] & windows[2])
        for p in parts:
            self.assertIn("DoS", set(frame.iloc[p]["_attack_type"]))
        self.assertEqual(sum(map(len, parts)), len(frame))


class PreprocessingTests(unittest.TestCase):
    def test_identifiers_labels_and_ports_are_not_raw_features(self):
        frame = pd.DataFrame({
            "Source IP": ["10.0.0.1", "10.0.0.2"], "Destination IP": ["10.0.0.2", "10.0.0.1"],
            "Source Port": [80, 51000], "Destination Port": [443, 80], "Timestamp": ["x", "y"],
            "Label": ["BENIGN", "DoS"], "Protocol": [6, 6], "Flow Bytes/s": [100.0, 1_000_000.0],
            "_window": [0, 1], "_day": ["Monday", "Monday"],
        })
        prep = FeaturePreprocessor(engineered=True).fit(frame.iloc[:1])
        x = prep.transform(frame)
        names = prep.output_names
        self.assertFalse(any("IP" in n or n in {"Source Port", "Destination Port", "Label", "Timestamp"} for n in names))
        self.assertFalse(any(n.startswith("_") for n in names))
        self.assertTrue(any("port_bucket" in n for n in names))
        self.assertTrue(np.isfinite(x).all())


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.frame = toy_frame()
        self.features = np.arange(12, dtype=np.float32).reshape(6, 2) / 10

    def test_one_graph_per_window_with_structural_features(self):
        graphs = build_graph_windows(self.frame, self.features)
        self.assertEqual(len(graphs), 2)
        g1 = graphs[1]  # host a contacts d and e on two ports: a small scan
        self.assertEqual(g1.edge_index.shape, (2, 4))
        self.assertEqual(g1.struct_x.shape[1], N_STRUCT)
        self.assertEqual(g1.node_x.shape[1], N_STRUCT + 2)
        a = sorted(set(g1.src_hosts) | set(g1.dst_hosts)).index("a")
        self.assertAlmostEqual(float(g1.struct_x[a, 2]), float(np.log1p(2)), places=5)  # distinct destinations
        self.assertAlmostEqual(float(g1.struct_x[a, 4]), float(np.log1p(2)), places=5)  # distinct dst ports
        np.testing.assert_array_equal(flow_order(graphs), np.arange(6))

    def test_models_return_one_output_per_flow_and_ae_ignores_target_features(self):
        graph = build_graph_windows(self.frame, self.features)[0]
        for use_flow in (True, False):
            logits = EdgeGraphSAGE(flow_dim=2, hidden=8, use_flow_features=use_flow).eval()(graph)
            self.assertEqual(tuple(logits.shape), (4, 2))
        ae = EdgeFeatureAutoencoder(edge_dim=2, hidden=8).eval()
        with torch.no_grad():
            before = ae(graph)
            graph.edge_x = torch.full_like(graph.edge_x, 99.0)
            after = ae(graph)
        self.assertTrue(torch.allclose(before, after))

    def test_autoencoder_training_graph_contains_only_benign_edges(self):
        graph = build_graph_windows(self.frame, self.features)[0]
        benign = _benign_only_graph(graph)
        self.assertEqual(benign.y.tolist(), [0, 0])
        self.assertEqual(benign.edge_index.shape[1], 4)
        self.assertEqual(benign.struct_x.shape, graph.struct_x.shape)

    def test_label_budget_is_class_stratified(self):
        y = torch.tensor([0] * 10 + [1] * 4)
        mask = _labeled_mask(y, .5, np.random.default_rng(7))
        self.assertEqual(torch.bincount(y[mask], minlength=2).tolist(), [5, 2])

    def test_trainers_run_with_validation_and_early_stopping(self):
        graphs = build_graph_windows(self.frame, self.features)
        gnn = fit_supervised(graphs, 2, seed=11, epochs=3, hidden=8, validation_graphs=graphs, patience=1)
        ae = fit_benign_autoencoder(graphs, 2, seed=11, epochs=3, hidden=8, validation_graphs=graphs, patience=1)
        self.assertTrue(1 <= gnn.epochs_trained <= 3 and 1 <= ae.epochs_trained <= 3)


class MetricTests(unittest.TestCase):
    def test_f1_threshold_sits_inside_a_flat_plateau_not_at_its_edge(self):
        y = np.array([0] * 50 + [1] * 50)
        s = np.concatenate([np.linspace(0, .2, 50), np.linspace(.8, 1, 50)])  # any threshold in (0.2, 0.8] is perfect
        t = threshold_max_f1(y, s)
        self.assertTrue(.2 < t <= .9)

    def test_thresholds_use_validation_only(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        s = np.array([.1, .2, .3, .4, .35, .9])
        self.assertLessEqual(threshold_max_f1(y, s), .35)
        t = threshold_at_fpr(y, s, fpr=0.0)
        self.assertGreater(t, .4)
        m = classification_metrics(y, s, np.array(["Benign"] * 4 + ["DoS", "DoS"]), t)
        self.assertEqual(m["false_positive_rate"], 0.0)
        self.assertEqual(m["detection_rate_per_attack"]["DoS"]["detection_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
