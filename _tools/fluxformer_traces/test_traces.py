"""Integration and numerical regression tests using real local artifacts."""

import ast
import contextlib
import copy
import csv
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import export_traces as exporter


def load_json(path):
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def resolve(document, value, common):
    """Resolve the documented cross-file JSON Pointers, including chained refs."""
    while "$ref" in value:
        filename, pointer = value["$ref"].split("#", 1)
        value = common if filename == "common.json.gz" else document
        for part in pointer.lstrip("/").split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
    tensor = torch.tensor(value["data"], dtype=getattr(torch, value["dtype"]))
    if list(tensor.shape) != value["shape"] or tensor.ndim != len(value["axes"]):
        raise AssertionError("Invalid serialized tensor shape/axes")
    return tensor


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.repo = Path(__file__).resolve().parents[2].parent / "metabolic-NN"
        cls.bundle = exporter.load_bundle(cls.repo)
        cls.temporary = tempfile.TemporaryDirectory(prefix="fluxformer-verification-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name)
        cls.manifest = exporter.export(cls.bundle, cls.output)
        cls.common = load_json(cls.output / "common.json.gz")
        cls.documents = [load_json(cls.output / example["file"]) for example in cls.manifest["examples"]]

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=exporter.ATOL, rtol=exporter.RTOL)

    def test_training_preprocessing_and_token_mapping(self):
        # Execute only the actual trainer's loader and its helper, avoiding all
        # unrelated imports/training code. Compare its real memmap output.
        source = ast.parse((self.repo / "train_flux_transformer.py").read_text())
        nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_count_rows", "load_data")]
        namespace = {"pd": pd, "np": np, "os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_trainer_loader", "exec"), namespace)
        records = [doc["sample"]["source_record"] for doc in self.documents]
        # Also check the original loader's missing-input zero initialization.
        missing = dict(records[0])
        missing[self.bundle["inputs"][0]] = ""
        records.append(missing)
        missing_nan = dict(records[0])
        missing_nan[self.bundle["inputs"][1]] = "NaN"
        records.append(missing_nan)
        csv_path = self.output / "loader-fixture.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.bundle["inputs"] + self.bundle["outputs"])
            writer.writeheader()
            writer.writerows(records)
        with contextlib.redirect_stdout(io.StringIO()):
            x, y, inputs, outputs, indices, _ = namespace["load_data"](
                csv_path, cache_dir=self.output / "memmap", chunksize=2)
        self.assertEqual(inputs, self.bundle["inputs"])
        self.assertEqual(outputs, self.bundle["outputs"])
        self.assertEqual(indices, self.bundle["indices"])
        for row, record in enumerate(records):
            c, _, target = exporter.input_tensors(self.bundle, record)
            self.assertTrue(torch.equal(c[0, :, 0], torch.from_numpy(x[row].copy())))
            self.assertTrue(torch.equal(target, torch.from_numpy(y[row].copy())))

    def test_original_trainer_and_canonical_forward_agree(self):
        classes = exporter.source_classes(self.repo / "train_flux_transformer.py")
        namespace = {"torch": torch, "nn": nn, "F": F}
        exec(compile(ast.Module(body=list(classes.values()), type_ignores=[]),
                     "original_trainer_classes", "exec"), namespace)
        config = self.bundle["checkpoint"]["config"]
        original = namespace["FluxTransformer"](**{key: config[key] for key in
            ("vocab_size", "input_token_indices", "d_model", "n_heads", "n_layers", "d_ff", "dropout")})
        original.load_state_dict(self.bundle["checkpoint"]["model_state_dict"], strict=True)
        original.eval()
        with torch.inference_mode():
            for doc in self.documents:
                c, _, _ = exporter.input_tensors(self.bundle, doc["sample"]["source_record"])
                actual, indices = self.bundle["model"](c, output_subset=None)
                expected, expected_indices = original(c, output_subset=None)
                self.assertTrue(torch.equal(indices, expected_indices))
                self.assertTrue(torch.equal(actual, expected))

    def validate_document(self, doc):
        """Re-read JSON and compare directly with fresh original-model outputs."""
        get = lambda value: resolve(doc, value, self.common)
        c, constraints, targets = exporter.input_tensors(self.bundle, doc["sample"]["source_record"])
        self.assertTrue(torch.equal(get(doc["initial_c"]), c[0]))
        self.assertTrue(torch.equal(get(doc["constraints"]), constraints))
        self.assertTrue(torch.equal(get(doc["targets"]), targets))
        self.assertTrue(torch.equal(get(doc["initial_x"]), self.bundle["model"].input_embedding.weight))
        captured, prediction = exporter.capture_forward(self.bundle["model"], c)
        for layer in doc["layers"]:
            i = layer["index"]
            prefix = f"layers.{i}"
            operations = layer["operations"]
            values = {name: get(value) for name, value in operations.items()}
            x, input_c = captured[prefix][0]
            self.assertClose(get(layer["input_x"]), x[0])
            self.assertClose(get(layer["input_c"]), input_c[0])
            block = self.bundle["model"].layers[i].attention_block
            norm_x = captured[f"{prefix}.attention_block.layer_norm"][1]
            self.assertClose(values["attention_norm"], norm_x[0])
            with torch.inference_mode():
                qkv = F._in_projection_packed(norm_x, norm_x, norm_x,
                                               block.mha.in_proj_weight, block.mha.in_proj_bias)
            for name, expected in zip(("query", "key", "value"), qkv):
                self.assertClose(values[name], expected[0].reshape(95, 8, 16).transpose(0, 1))
            expected_output, expected_attention = captured[f"{prefix}.attention_block.mha"][1]
            self.assertEqual(values["attention"].shape, (8, 95, 95))
            self.assertClose(values["attention"], expected_attention[0])
            # Softmax inversion checks the exported logits against native attention;
            # float64 logits contraction is separately checked during generation.
            self.assertClose(values["scaled_logits"].softmax(-1), expected_attention[0])
            self.assertClose(values["weighted_values"], expected_attention[0] @ values["value"])
            self.assertClose(values["combined_heads"], values["weighted_values"].transpose(0, 1).reshape(95, 128))
            self.assertClose(values["output_projection"], expected_output[0])
            self.assertClose(values["embedding_residual"], x[0] + expected_output[0])
            self.assertClose(values["head_weights"], block.head_scores.detach().softmax(0))
            scalar_heads = expected_attention[0] @ input_c[0]
            self.assertClose(values["scalar_heads"], scalar_heads)
            mixed = torch.einsum("h,hic->ic", block.head_scores.detach().softmax(0), scalar_heads)
            self.assertClose(values["scalar_combined"], mixed)
            self.assertClose(values["scalar_residual"], captured[f"{prefix}.attention_block"][1][1][0])
            self.assertClose(values["concatenated"], captured[f"{prefix}.feedforward_block.layer_norm"][0][0][0])
            for name, module in (("feedforward_norm", "layer_norm"), ("linear1", "linear1"),
                                 ("gelu", "activation"), ("dropout", "dropout"), ("linear2", "linear2")):
                self.assertClose(values[name], captured[f"{prefix}.feedforward_block.{module}"][1][0])
            self.assertClose(values["feedforward_residual"], captured[f"{prefix}.feedforward_block"][1][0])
            self.assertClose(values["output_x"], captured[prefix][1][0][0])
            self.assertClose(values["output_c"][:, 0], prediction[0, :, i])
        self.assertClose(get(doc["final_predictions"])[:, 0], prediction[0, :, -1])

    def test_serialized_operations_against_independent_original_forward(self):
        with torch.inference_mode():
            for doc in self.documents:
                with self.subTest(row=doc["sample"]["csv_data_row_zero_based"]):
                    self.validate_document(doc)

    def test_verification_covers_all_layers_and_references(self):
        for doc in self.documents:
            report = doc["verification"]
            self.assertTrue(report["passed"])
            operations = {check["operation"] for check in report["checks"]}
            for i in range(3):
                for name in ("query", "key", "value", "scaled_logits", "attention_native",
                             "attention_functional", "attention_sdpa", "weighted_values",
                             "output_projection_native", "scalar_heads", "head_weights",
                             "scalar_residual", "embedding_residual", "concatenation",
                             "feedforward_norm", "linear1", "gelu", "eval_dropout", "linear2",
                             "feedforward_residual", "output_x", "output_c", "model_intermediate"):
                    self.assertIn(f"layers.{i}.{name}", operations)
            self.assertIn("all_layer_predictions_vs_fresh_forward", operations)
            self.assertIn("final_reconstructed_fluxes_vs_original_forward", operations)
            for i in range(3):
                self.assertIn(f"layers.{i}.end_to_end_reconstruction_x", operations)
                self.assertIn(f"layers.{i}.end_to_end_reconstruction_c", operations)
            self.assertTrue(all(check["max_tolerance_ratio"] <= 1 for check in report["checks"]))

    def test_corrupted_export_is_detected(self):
        corrupted = copy.deepcopy(self.documents[0])
        corrupted["layers"][1]["operations"]["attention"]["data"][0][0][0] += 0.01
        with self.assertRaises(AssertionError), torch.inference_mode():
            self.validate_document(corrupted)
        corrupted = copy.deepcopy(self.documents[0])
        corrupted["layers"][-1]["operations"]["output_c"]["data"][12][0] += 1.0
        with self.assertRaises(AssertionError), torch.inference_mode():
            self.validate_document(corrupted)

    def test_input_failures_are_explicit(self):
        bad = dict(self.documents[0]["sample"]["source_record"])
        bad[self.bundle["inputs"][0]] = "inf"
        with self.assertRaisesRegex(ValueError, "non-finite"):
            exporter.input_tensors(self.bundle, bad)
        bad[self.bundle["inputs"][0]] = "10"
        with self.assertRaisesRegex(ValueError, "Positive bounds"):
            exporter.input_tensors(self.bundle, bad)
        with self.assertRaisesRegex(ValueError, "outside"):
            exporter.select_samples(self.bundle, rows=[100000])

    def test_first_layer_attention_independent_but_later_states_change(self):
        attention = []
        with torch.inference_mode():
            for doc in self.documents:
                c, _, _ = exporter.input_tensors(self.bundle, doc["sample"]["source_record"])
                captured, _ = exporter.capture_forward(self.bundle["model"], c)
                attention.append(captured["layers.0.attention_block.mha"][1][1])
        self.assertTrue(all(torch.equal(value, attention[0]) for value in attention[1:]))
        first, second = self.documents[:2]
        get = lambda doc, i, name: resolve(doc, doc["layers"][i]["operations"][name], self.common)
        self.assertFalse(torch.equal(get(first, 0, "output_c"), get(second, 0, "output_c")))
        self.assertFalse(torch.equal(get(first, 1, "attention"), get(second, 1, "attention")))

    def test_export_integrity_and_float32_json_roundtrip(self):
        self.assertEqual(len(self.common["reactions"]), 95)
        self.assertEqual(len(self.common["inputs"]), 20)
        self.assertEqual(self.manifest["common_sha256"], exporter.sha256(self.output / "common.json.gz"))
        for example in self.manifest["examples"]:
            self.assertEqual(example["sha256"], exporter.sha256(self.output / example["file"]))
        for name, expected in self.bundle["model"].state_dict().items():
            actual = resolve({}, self.common["parameters"][name], self.common)
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
