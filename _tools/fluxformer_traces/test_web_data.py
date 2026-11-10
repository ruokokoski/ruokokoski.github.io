"""Check every published value against the verified source traces, without Torch."""

import json
from pathlib import Path
import tempfile
import unittest

import prepare_web_data as packager


class PublishedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).parent / "examples"
        cls.public = packager.ROOT / "data/fluxformer"
        cls.original_manifest = packager.read(cls.source / "manifest.json")
        cls.common = packager.read(cls.source / "common.json.gz")
        cls.meta = packager.read(cls.public / "manifest.json")

    def test_metadata_initialization_and_projection_weights_are_lossless(self):
        self.assertEqual(self.meta["reactions"], self.common["reactions"])
        self.assertEqual(self.meta["inputs"], self.common["inputs"])
        self.assertEqual(self.meta["initial_x"], self.common["parameters"]["input_embedding.weight"]["data"])
        for i in range(3):
            self.assertEqual(self.meta["head_weights"][i], self.common["head_weights"][i]["data"])
            self.assertEqual(self.meta["scalar_projection"][i]["weights"],
                             self.common["parameters"][f"layers.{i}.feedforward_block.linear2.weight"]["data"][-1])

    def test_every_published_layer_operation_matches_all_95_source_tokens(self):
        for sample_meta, original_meta in zip(self.meta["samples"], self.original_manifest["examples"]):
            original = packager.read(self.source / original_meta["file"])
            for i, filename in enumerate(sample_meta["layers"]):
                with self.subTest(sample=sample_meta["row"], layer=i):
                    layer = packager.read(self.public / filename)
                    expected = original["layers"][i]
                    for key in ("input_x", "input_c"):
                        self.assertEqual(layer[key], packager.resolve(original, expected[key], self.common))
                    self.assertEqual(set(layer["operations"]), set(expected["operations"]))
                    for name, operation in expected["operations"].items():
                        self.assertEqual(layer["operations"][name], packager.resolve(original, operation, self.common), name)
                    self.assertEqual(len(layer["operations"]["attention"]), 8)
                    self.assertTrue(all(len(row) == 95 for head in layer["operations"]["attention"] for row in head))
                    self.assertEqual(packager.digest(self.public / filename), sample_meta["files_sha256"][filename])

    def test_all_overview_bounds_targets_and_layer_states_are_exact(self):
        for sample_meta, original_meta in zip(self.meta["samples"], self.original_manifest["examples"]):
            original = packager.read(self.source / original_meta["file"])
            overview = packager.read(self.public / sample_meta["overview"])
            self.assertEqual(overview["constraints"], original["constraints"]["data"])
            self.assertEqual(overview["targets"], original["targets"]["data"])
            self.assertEqual(overview["initial_c"], [value[0] for value in original["initial_c"]["data"]])
            for i in range(3):
                self.assertEqual(overview["states"][i], [value[0] for value in packager.resolve(
                    original, original["layers"][i]["operations"]["output_c"], self.common)])
            self.assertEqual(sample_meta["source_sha256"], packager.digest(self.source / original_meta["file"]))
            self.assertEqual(packager.digest(self.public / sample_meta["overview"]),
                             sample_meta["files_sha256"][sample_meta["overview"]])

    def test_packager_rejects_unverified_or_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = json.loads(json.dumps(self.original_manifest))
            manifest["verification"]["passed"] = False
            packager.write(path / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "verified trace"):
                packager.prepare(path, path / "out")
            manifest["verification"]["passed"] = True
            packager.write(path / "manifest.json", manifest)
            (path / manifest["common"]).write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                packager.prepare(path, path / "out")


if __name__ == "__main__":
    unittest.main()
