"""Package verified traces for static, lazy-loaded presentation; no inference."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS = {870: "Low simulated growth", 722: "High simulated growth",
          962: "Mixed nutrients A", 63: "Mixed nutrients B"}


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(document, value, common):
    while "$ref" in value:
        filename, pointer = value["$ref"].split("#", 1)
        value = common if filename == "common.json.gz" else document
        for part in pointer.lstrip("/").split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
    return value["data"]


def write(path, data):
    raw = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def prepare(source, output):
    source, output = Path(source), Path(output)
    manifest = read(source / "manifest.json")
    if manifest["schema_version"] != "1.0.0" or not manifest["verification"]["passed"]:
        raise ValueError("Expected verified trace schema 1.0.0")
    if digest(source / manifest["common"]) != manifest["common_sha256"]:
        raise ValueError("Shared trace hash mismatch")
    common = read(source / manifest["common"])
    if not common["first_layer_independence"]["passed"]:
        raise ValueError("First-layer independence has not been verified")
    output.mkdir(parents=True, exist_ok=True)
    config = common["model"]["config"]
    if (config["vocab_size"], len(common["inputs"]), config["n_layers"]) != (95, 20, 3):
        raise ValueError("Unexpected model contract")
    meta = {
        "schema_version": "1.0.0", "model": common["model"],
        "reactions": common["reactions"], "inputs": common["inputs"],
        "initial_x": common["parameters"]["input_embedding.weight"]["data"],
        "head_weights": [value["data"] for value in common["head_weights"]],
        "scalar_projection": [{
            "weights": common["parameters"][f"layers.{i}.feedforward_block.linear2.weight"]["data"][-1],
            "bias": common["parameters"][f"layers.{i}.feedforward_block.linear2.bias"]["data"][-1],
        } for i in range(config["n_layers"])],
        "provenance": common["provenance"],
        "verification": manifest["verification"],
        "source_common_sha256": manifest["common_sha256"],
        "samples": [],
    }
    for example in manifest["examples"]:
        if digest(source / example["file"]) != example["sha256"]:
            raise ValueError(f"Sample hash mismatch: {example['file']}")
        sample = read(source / example["file"])
        checks = sample["verification"]["checks"]
        if not sample["verification"]["passed"] or not all(check["passed"] for check in checks):
            raise ValueError("Unverified sample")
        row = example["row"]
        prefix = f"sample-{row:06d}"
        overview = {
            "row": row, "constraints": sample["constraints"]["data"],
            "initial_c": [value[0] for value in sample["initial_c"]["data"]],
            "targets": sample["targets"]["data"],
            "states": [[value[0] for value in resolve(sample, layer["operations"]["output_c"], common)]
                       for layer in sample["layers"]],
            "verification": sample["verification"],
        }
        write(output / f"{prefix}.json", overview)
        layer_files = []
        for layer in sample["layers"]:
            filename = f"{prefix}-layer-{layer['index']}.json.gz"
            # All 95 query/source reactions and every recorded operation remain
            # intact. Only wrappers/references are removed; no values are rounded.
            write(output / filename, {
                "sample_row": row, "index": layer["index"], "state_kind": layer["state_kind"],
                "input_x": resolve(sample, layer["input_x"], common),
                "input_c": resolve(sample, layer["input_c"], common),
                "operations": {name: resolve(sample, value, common)
                               for name, value in layer["operations"].items()},
            })
            layer_files.append(filename)
        files = [f"{prefix}.json", *layer_files]
        meta["samples"].append({
            "row": row, "label": LABELS.get(row, f"Simulated condition {row}"),
            "overview": files[0], "layers": layer_files, "source_sha256": example["sha256"],
            "files_sha256": {name: digest(output / name) for name in files},
            "selection_reason": example["selection_reason"],
        })
    write(output / "manifest.json", meta)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "examples")
    parser.add_argument("--output", type=Path, default=ROOT / "data/fluxformer")
    args = parser.parse_args()
    data = prepare(args.source, args.output)
    print(f"Packaged {len(data['samples'])} verified conditions, 95 reactions, all three layers in {args.output}")
