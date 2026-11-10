"""Offline, full-vocabulary FluxFormer traces; no changes to the source model."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import platform
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn.functional as F


MODEL_NAME = "ecoli_core_10M_signed_pfba_d128_h8_l3_ff640"
DATA_NAME = "2026-10-13_ecoli_core_test_data_100000_samples.csv"
CLASSES = ("AttentionBlock", "FeedForwardBlock", "FluxTransformerLayer", "FluxTransformer")
ATOL = 2e-5
RTOL = 2e-5
SCHEMA_VERSION = "1.0.0"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_classes(path):
    return {
        node.name: node for node in ast.parse(Path(path).read_text()).body
        if isinstance(node, ast.ClassDef) and node.name in CLASSES
    }


def check_architecture(repo):
    canonical = source_classes(repo / "flux_transformer.py")
    trainer = source_classes(repo / "train_flux_transformer.py")
    matches = {
        name: name in canonical and name in trainer and
        ast.dump(canonical[name], include_attributes=False) ==
        ast.dump(trainer[name], include_attributes=False)
        for name in CLASSES
    }
    if not all(matches.values()):
        raise ValueError(f"Canonical/trainer architecture mismatch: {matches}")
    return matches


def load_bundle(repo):
    repo = Path(repo).resolve()
    checkpoint_path = repo / "models" / MODEL_NAME / f"{MODEL_NAME}_checkpoint.pth"
    dataset_path = repo / "data" / DATA_NAME
    sbml_path = repo / "models/e_coli_core.xml"
    log_path = checkpoint_path.with_name(f"{MODEL_NAME}_training.log")
    for path in (checkpoint_path, dataset_path, sbml_path, log_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required real artifact unavailable: {path}")
    matches = check_architecture(repo)
    # This is the user's trusted local training checkpoint, which includes NumPy
    # RNG objects as well as tensors. Never load untrusted pickle checkpoints.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, info = checkpoint["config"], checkpoint["data_info"]
    inputs, outputs = info["input_cols"], info["output_cols"]
    indices = [outputs.index(f"{name}_flux") for name in inputs]
    if len(inputs) != 20 or len(outputs) != 95 or len(set(outputs)) != 95:
        raise ValueError("Expected the exact 20-input / 95-token core contract")
    if indices != info["input_token_indices"] or indices != config["input_token_indices"]:
        raise ValueError("Checkpoint input mapping disagrees with its vocabulary")
    state = checkpoint["model_state_dict"]
    if state["input_token_indices"].tolist() != indices or config["vocab_size"] != 95:
        raise ValueError("Checkpoint state/config mapping mismatch")
    if any(value.dtype != torch.float32 for value in state.values() if value.is_floating_point()):
        raise ValueError("This exporter requires the checkpoint's original float32 weights")
    with dataset_path.open(newline="") as stream:
        if next(csv.reader(stream)) != inputs + outputs:
            raise ValueError("Test CSV column order differs from the training checkpoint")
    training_path = repo / info["dataset"]
    if not training_path.is_file():
        raise FileNotFoundError(f"Training CSV needed to verify its header: {training_path}")
    with training_path.open(newline="") as stream:
        if next(csv.reader(stream)) != inputs + outputs:
            raise ValueError("Training CSV column order differs from the checkpoint")
    spec = importlib.util.spec_from_file_location("fluxformer_trace_source", repo / "flux_transformer.py")
    module = importlib.util.module_from_spec(spec)
    # Compile the authoritative source without creating bytecode in the sibling repo.
    source_path = repo / "flux_transformer.py"
    exec(compile(source_path.read_text(), str(source_path), "exec"), module.__dict__)
    model = module.FluxTransformer(**{
        key: config[key] for key in
        ("vocab_size", "input_token_indices", "d_model", "n_heads", "n_layers", "d_ff", "dropout")
    })
    model.load_state_dict(state, strict=True)
    model.eval()
    root = ET.parse(sbml_path).getroot()
    namespace = root.tag.split("}")[0] + "}"
    sbml_names = {
        node.attrib["id"].removeprefix("R_"): node.attrib.get("name")
        for node in root.iter(f"{namespace}reaction")
    }
    token_ids = {token.removesuffix("_flux") for token in outputs}
    # The repository XML has two different IDs (FORt and BIOMASS_*). Do not
    # silently alias them to the trained textbook vocabulary.
    names = {name: sbml_names.get(name) for name in token_ids}
    name_artifact = sbml_path
    cobra_spec = importlib.util.find_spec("cobra")
    if cobra_spec is not None and cobra_spec.submodule_search_locations:
        textbook = Path(next(iter(cobra_spec.submodule_search_locations))) / "data/textbook.xml.gz"
        if textbook.is_file():
            with gzip.open(textbook, "rb") as stream:
                textbook_root = ET.parse(stream).getroot()
            textbook_names = {
                node.attrib["id"].removeprefix("R_"): node.attrib.get("name")
                for node in textbook_root.iter() if node.tag.endswith("}reaction")
            }
            if set(textbook_names) != token_ids:
                raise ValueError("Installed textbook model differs from the trained reaction vocabulary")
            names, name_artifact = textbook_names, textbook
    return {
        "repo": repo, "checkpoint_path": checkpoint_path, "dataset_path": dataset_path,
        "sbml_path": sbml_path, "log_path": log_path, "checkpoint": checkpoint,
        "model": model, "inputs": inputs, "outputs": outputs, "indices": indices,
        "reaction_names": names, "architecture_matches": matches,
        "name_artifact": name_artifact,
        "repository_sbml_only_ids": sorted(set(sbml_names) - token_ids),
        "checkpoint_only_ids_vs_repository_sbml": sorted(token_ids - set(sbml_names)),
    }


def input_tensors(bundle, record):
    """Match the trainer: raw signed float32 constraints, no scaling/sign flip."""
    constraints = torch.tensor([
        float(record[name]) if record[name].strip() else 0.0 for name in bundle["inputs"]
    ], dtype=torch.float32)
    constraints[torch.isnan(constraints)] = 0.0  # trainer: chunk[inputs].fillna(0.0)
    targets = torch.tensor([float(record[name]) for name in bundle["outputs"]], dtype=torch.float32)
    if not torch.isfinite(constraints).all() or not torch.isfinite(targets).all():
        raise ValueError("Sample has non-finite constraints or simulation targets")
    if (constraints > 0).any():
        raise ValueError("Positive bounds conflict with the selected signed-input dataset")
    c = torch.zeros((1, 95, 1), dtype=torch.float32)
    c[0, bundle["indices"], 0] = constraints
    return c, constraints, targets


def select_samples(bundle, rows=None, count=4, scan_rows=1024):
    """Deterministic illustrative selection, not an unbiased evaluation set."""
    if rows is not None and (not rows or len(set(rows)) != len(rows) or any(row < 0 for row in rows)):
        raise ValueError("Sample row indices must be unique and nonnegative")
    records = []
    limit = max(rows) + 1 if rows else scan_rows
    with bundle["dataset_path"].open(newline="") as stream:
        for index, record in enumerate(csv.DictReader(stream)):
            if index >= limit:
                break
            if rows is None or index in rows:
                records.append((index, record))
    if rows is not None:
        found = dict(records)
        if any(row not in found for row in rows):
            raise ValueError("Requested sample row is outside the real CSV")
        return [(row, found[row], "explicit zero-based CSV data row") for row in rows]
    if not 3 <= count <= 5 or len(records) < count:
        raise ValueError("Select 3–5 examples from a sufficiently large scan")
    features = torch.tensor([
        [float(record[name]) for name in bundle["inputs"]] + [float(record["Biomass_Ecoli_core_flux"])]
        for _, record in records
    ], dtype=torch.float64)
    selected = [int(features[:, -1].argmin()), int(features[:, -1].argmax())]
    if selected[1] == selected[0]:
        selected = selected[:1]
    reasons = ["lowest simulated biomass in scanned prefix", "highest simulated biomass in scanned prefix"]
    ranges = features.amax(0) - features.amin(0)
    normalized = (features - features.amin(0)) / ranges.clamp_min(1e-12)
    while len(selected) < count:
        distances = torch.cdist(normalized, normalized[selected]).square().amin(1)
        distances[selected] = -1
        selected.append(int(distances.argmax()))
        if len(selected) > len(reasons):
            reasons.append("farthest from selected conditions in range-scaled bounds + biomass")
    return [(records[index][0], records[index][1], reasons[i]) for i, index in enumerate(selected)]


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return tuple(clone_value(item) for item in value)


@torch.inference_mode()
def capture_forward(model, c):
    """Read-only hooks on the original forward, including native PyTorch MHA."""
    captured, handles = {}, []

    def hook(name):
        def save(module, args, output):
            captured[name] = (clone_value(args), clone_value(output))
        return save

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(hook(name)))
    try:
        prediction, indices = model(c, output_subset=None)
    finally:
        for handle in handles:
            handle.remove()
    if indices.tolist() != list(range(95)):
        raise AssertionError("Full-vocabulary inference did not retain all 95 reactions")
    return captured, prediction.detach().clone()


class Verification:
    def __init__(self):
        self.checks = []

    def compare(self, name, actual, reference):
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise AssertionError(f"{name}: shape or dtype mismatch")
        if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
            raise AssertionError(f"{name}: non-finite result")
        difference = (actual.double() - reference.double()).abs()
        denominator = reference.double().abs()
        check = {
            "operation": name, "shape": list(actual.shape),
            "max_absolute_error": difference.max().item(),
            "max_relative_error": (difference / denominator.clamp_min(1e-7)).max().item(),
            "max_tolerance_ratio": (difference / (ATOL + RTOL * denominator)).max().item(),
            "passed": bool(torch.all(difference <= ATOL + RTOL * denominator)),
        }
        self.checks.append(check)
        if not check["passed"]:
            raise AssertionError(f"Numerical verification failed: {check}")

    def report(self):
        return {
            "passed": all(check["passed"] for check in self.checks),
            "atol": ATOL, "rtol": RTOL, "relative_error_denominator_floor": 1e-7,
            "checks": self.checks,
        }


def tensor_json(tensor, axes):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.ndim != len(axes):
        raise ValueError("Tensor axes do not match its rank")
    return {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape),
            "axes": axes, "data": tensor.tolist()}


@torch.inference_mode()
def reconstruct_forward(model, c):
    """End-to-end explicit calculation, never calling a model block or MHA."""
    x = model.input_embedding.weight.unsqueeze(0).clone()
    c = c.clone()
    states = []
    for layer in model.layers:
        block, ff = layer.attention_block, layer.feedforward_block
        normalized = F.layer_norm(x, (model.d_model,), block.layer_norm.weight,
                                  block.layer_norm.bias, block.layer_norm.eps)
        q, k, v = [value.reshape(1, 95, block.n_heads, block.mha.head_dim).transpose(1, 2)
                   for value in F.linear(normalized, block.mha.in_proj_weight,
                                         block.mha.in_proj_bias).chunk(3, -1)]
        attention = (torch.matmul(q * math.sqrt(1.0 / block.mha.head_dim),
                                  k.transpose(-2, -1))).softmax(-1)
        combined = (attention @ v).transpose(1, 2).contiguous().reshape(1, 95, model.d_model)
        x = x + F.linear(combined, block.mha.out_proj.weight, block.mha.out_proj.bias)
        scalar_heads = attention @ c.unsqueeze(1)
        c = c + (scalar_heads * block.head_scores.softmax(0).view(1, -1, 1, 1)).sum(1)
        joined = torch.cat((x, c), -1)
        normalized = F.layer_norm(joined, (model.d_model + 1,), ff.layer_norm.weight,
                                  ff.layer_norm.bias, ff.layer_norm.eps)
        hidden = F.gelu(F.linear(normalized, ff.linear1.weight, ff.linear1.bias))
        joined = joined + F.linear(hidden, ff.linear2.weight, ff.linear2.bias)
        x, c = joined[:, :, :-1], joined[:, :, -1:]
        states.append((x, c))
    return states


@torch.inference_mode()
def trace_sample(bundle, record):
    model = bundle["model"]
    if model.training:
        raise ValueError("Tracing requires model.eval()")
    c, constraints, targets = input_tensors(bundle, record)
    captured, original_prediction = capture_forward(model, c)
    verification = Verification()
    initial_x = captured["input_embedding"][1]
    verification.compare("initial_embeddings", initial_x[0], model.input_embedding.weight)
    expected_c = torch.tensor([
        float(record[token.removesuffix("_flux")]) if token.removesuffix("_flux") in bundle["inputs"]
        and record[token.removesuffix("_flux")].strip() else 0.0 for token in bundle["outputs"]
    ], dtype=torch.float32)
    expected_c[torch.isnan(expected_c)] = 0.0
    verification.compare("initial_scalar_mapping", c[0, :, 0], expected_c)
    layers = []
    for index, layer in enumerate(model.layers):
        prefix = f"layers.{index}"
        block, ff = layer.attention_block, layer.feedforward_block
        x, current_c = captured[prefix][0]
        norm_x = captured[f"{prefix}.attention_block.layer_norm"][1]
        # Independent float64 reduction verifies LayerNorm, not another call to it.
        def reference_norm(value, module):
            high = value.double()
            mean = high.mean(-1, keepdim=True)
            variance = ((high - mean) ** 2).mean(-1, keepdim=True)
            return (((high - mean) / (variance + module.eps).sqrt()) *
                    module.weight.double() + module.bias.double()).float()

        verification.compare(f"{prefix}.attention_norm", norm_x, reference_norm(x, block.layer_norm))
        mha = block.mha
        if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
            raise ValueError("Unexpected attention options in the core checkpoint")
        # Packed Q/K/V projection is the actual nn.MultiheadAttention layout.
        packed = F.linear(norm_x, mha.in_proj_weight, mha.in_proj_bias)
        projections = packed.chunk(3, dim=-1)
        reference_qkv = F._in_projection_packed(norm_x, norm_x, norm_x,
                                               mha.in_proj_weight, mha.in_proj_bias)
        head_dim = mha.head_dim
        q, k, v = [value.reshape(1, 95, mha.num_heads, head_dim).transpose(1, 2)
                   for value in projections]
        for name, actual, reference in zip(("query", "key", "value"), projections, reference_qkv):
            verification.compare(f"{prefix}.{name}", actual, reference)
        logits = torch.matmul(q * math.sqrt(1.0 / head_dim), k.transpose(-2, -1))
        # Independent contraction in float64, rounded back to the original dtype.
        reference_logits = (torch.einsum("bhid,bhjd->bhij", q.double(), k.double()) /
                            math.sqrt(head_dim)).float()
        verification.compare(f"{prefix}.scaled_logits", logits, reference_logits)
        reconstructed_attention = logits.softmax(-1)
        native_output, attention = captured[f"{prefix}.attention_block.mha"][1]
        verification.compare(f"{prefix}.attention_native", reconstructed_attention, attention)
        # Separate public functional PyTorch reference takes the unfused MHA path.
        functional_output, functional_attention = F.multi_head_attention_forward(
            norm_x.transpose(0, 1), norm_x.transpose(0, 1), norm_x.transpose(0, 1),
            mha.embed_dim, mha.num_heads, mha.in_proj_weight, mha.in_proj_bias,
            mha.bias_k, mha.bias_v, mha.add_zero_attn, mha.dropout,
            mha.out_proj.weight, mha.out_proj.bias, training=False,
            need_weights=True, average_attn_weights=False,
        )
        verification.compare(f"{prefix}.attention_functional", attention, functional_attention)
        weighted_v = torch.matmul(attention, v)
        # SDPA with an identity-valued V exposes attention through a separate kernel.
        identity = torch.eye(95, dtype=torch.float32).expand(1, mha.num_heads, 95, 95)
        sdpa_attention = F.scaled_dot_product_attention(q, k, identity, dropout_p=0.0)
        verification.compare(f"{prefix}.attention_sdpa", attention, sdpa_attention)
        verification.compare(f"{prefix}.weighted_values", weighted_v,
                             F.scaled_dot_product_attention(q, k, v, dropout_p=0.0))
        combined = weighted_v.transpose(1, 2).contiguous().reshape(1, 95, mha.embed_dim)
        projected = F.linear(combined, mha.out_proj.weight, mha.out_proj.bias)
        verification.compare(f"{prefix}.output_projection_native", projected, native_output)
        verification.compare(f"{prefix}.output_projection_functional", projected,
                             functional_output.transpose(0, 1))
        attn_x, attn_c = captured[f"{prefix}.attention_block"][1]
        verification.compare(f"{prefix}.embedding_residual", x + native_output, attn_x)
        scalar_heads = torch.matmul(attention, current_c.unsqueeze(1))
        alpha = block.head_scores.softmax(0)
        reference_alpha = torch.exp(block.head_scores.double() - block.head_scores.double().max())
        reference_alpha = (reference_alpha / reference_alpha.sum()).float()
        verification.compare(f"{prefix}.head_weights", alpha, reference_alpha)
        reference_scalar_heads = torch.einsum("bhij,bjc->bhic", attention, current_c)
        verification.compare(f"{prefix}.scalar_heads", scalar_heads, reference_scalar_heads)
        scalar_combined = (scalar_heads * alpha.view(1, -1, 1, 1)).sum(1)
        verification.compare(f"{prefix}.scalar_residual", scalar_combined + current_c, attn_c)
        joined = torch.cat((attn_x, attn_c), dim=-1)
        norm_y = captured[f"{prefix}.feedforward_block.layer_norm"][1]
        verification.compare(f"{prefix}.concatenation", joined,
                             captured[f"{prefix}.feedforward_block.layer_norm"][0][0])
        verification.compare(f"{prefix}.feedforward_norm", norm_y, reference_norm(joined, ff.layer_norm))
        first = captured[f"{prefix}.feedforward_block.linear1"][1]
        gelu = captured[f"{prefix}.feedforward_block.activation"][1]
        after_dropout = captured[f"{prefix}.feedforward_block.dropout"][1]
        second = captured[f"{prefix}.feedforward_block.linear2"][1]
        verification.compare(f"{prefix}.linear1", F.linear(norm_y, ff.linear1.weight, ff.linear1.bias), first)
        reference_gelu = (first.double() * 0.5 * (1 + torch.erf(first.double() / math.sqrt(2)))).float()
        verification.compare(f"{prefix}.gelu", gelu, reference_gelu)
        verification.compare(f"{prefix}.eval_dropout", gelu, after_dropout)
        verification.compare(f"{prefix}.linear2", F.linear(after_dropout, ff.linear2.weight, ff.linear2.bias), second)
        ff_residual = joined + second
        verification.compare(f"{prefix}.feedforward_residual", ff_residual,
                             captured[f"{prefix}.feedforward_block"][1])
        final_x, final_c = captured[prefix][1]
        verification.compare(f"{prefix}.output_x", ff_residual[:, :, :-1], final_x)
        verification.compare(f"{prefix}.output_c", ff_residual[:, :, -1:], final_c)
        verification.compare(f"{prefix}.model_intermediate", final_c[:, :, 0], original_prediction[:, :, index])
        operations = {}

        def add(name, value, axes):
            operations[name] = tensor_json(value[0], axes)

        add("attention_norm", norm_x, ["reaction", "embedding"])
        for name, value in (("query", q), ("key", k), ("value", v)):
            add(name, value, ["head", "reaction", "head_feature"])
        for name, value in (("scaled_logits", logits), ("attention", attention)):
            add(name, value, ["head", "query_reaction", "source_reaction"])
        add("weighted_values", weighted_v, ["head", "reaction", "head_feature"])
        add("combined_heads", combined, ["reaction", "embedding"])
        add("output_projection", native_output, ["reaction", "embedding"])
        add("embedding_residual", attn_x, ["reaction", "embedding"])
        add("scalar_heads", scalar_heads, ["head", "reaction", "scalar"])
        operations["head_weights"] = tensor_json(alpha, ["head"])
        add("scalar_combined", scalar_combined, ["reaction", "scalar"])
        add("scalar_residual", attn_c, ["reaction", "scalar"])
        for name, value, feature in (
            ("concatenated", joined, "joined_feature"), ("feedforward_norm", norm_y, "joined_feature"),
            ("linear1", first, "hidden_feature"), ("gelu", gelu, "hidden_feature"),
            ("linear2", second, "joined_feature"), ("feedforward_residual", ff_residual, "joined_feature"),
            ("output_x", final_x, "embedding"), ("output_c", final_c, "scalar"),
        ):
            add(name, value, ["reaction", feature])
        layers.append({"index": index, "state_kind": "intermediate" if index < len(model.layers)-1 else "final",
                       "operations": operations})
    traced_predictions = torch.stack([
        torch.tensor(layer["operations"]["output_c"]["data"], dtype=torch.float32)[:, 0]
        for layer in layers
    ], dim=-1).unsqueeze(0)
    # Fresh complete model.forward, with hooks removed, verifies every exported scalar.
    reference_prediction, _ = model(c, output_subset=None)
    verification.compare("all_layer_predictions_vs_fresh_forward", traced_predictions, reference_prediction)
    reconstructed = reconstruct_forward(model, c)
    for i, (reconstructed_x, reconstructed_c) in enumerate(reconstructed):
        actual_x, actual_c = captured[f"layers.{i}"][1]
        verification.compare(f"layers.{i}.end_to_end_reconstruction_x", reconstructed_x, actual_x)
        verification.compare(f"layers.{i}.end_to_end_reconstruction_c", reconstructed_c, actual_c)
    verification.compare("final_reconstructed_fluxes_vs_original_forward",
                         reconstructed[-1][1][:, :, 0], reference_prediction[:, :, -1])
    return {
        "initial_c": tensor_json(c[0], ["reaction", "scalar"]),
        "constraints": tensor_json(constraints, ["input"]),
        "targets": tensor_json(targets, ["reaction"]),
        "layers": layers, "verification": verification.report(),
    }


def write_json(path, value):
    """Deterministic gzip (zero timestamp) containing ordinary UTF-8 JSON."""
    content = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if path.suffix == ".gz":
        content = gzip.compress(content, compresslevel=6, mtime=0)
    path.write_bytes(content)


def reference(pointer):
    return {"$ref": pointer}


def export(bundle, output, rows=None, count=4, scan_rows=1024):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = select_samples(bundle, rows, count, scan_rows)
    traces = [trace_sample(bundle, record) for _, record, _ in samples]
    # The first-layer embedding path is condition-independent: prove this for
    # every selected real condition before storing it once in the common file.
    static_names = ("attention_norm", "query", "key", "value", "scaled_logits", "attention",
                    "weighted_values", "combined_heads", "output_projection", "embedding_residual")
    independence = Verification()
    for sample_index, trace in enumerate(traces[1:], 1):
        for name in static_names:
            a = torch.tensor(trace["layers"][0]["operations"][name]["data"], dtype=torch.float32)
            b = torch.tensor(traces[0]["layers"][0]["operations"][name]["data"], dtype=torch.float32)
            independence.compare(f"sample_{sample_index}.first_layer.{name}", a, b)
            if not torch.equal(a, b):
                raise AssertionError("Condition-independent first layer was not bitwise equal")
    repo = bundle["repo"]
    artifacts = {
        name: {"path": str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path), "sha256": sha256(path)}
        for name, path in {
            "checkpoint": bundle["checkpoint_path"], "test_csv": bundle["dataset_path"],
            "sbml": bundle["sbml_path"], "training_log": bundle["log_path"],
            "reaction_names": bundle["name_artifact"],
            "model_source": repo / "flux_transformer.py", "trainer_source": repo / "train_flux_transformer.py",
            "generator_source": repo / "generate_ecoli_core_data.py",
        }.items()
    }
    checkpoint = bundle["checkpoint"]
    common = {
        "schema_version": SCHEMA_VERSION, "kind": "common",
        "model": {"name": MODEL_NAME, "config": checkpoint["config"],
                  "model_epoch": checkpoint["model_epoch"], "best_epoch": checkpoint["best_epoch"],
                  "best_validation_loss": checkpoint["best_test_loss"],
                  "training_data_info": checkpoint["data_info"],
                  "layer_settings": [{
                      "attention_norm_eps": layer.attention_block.layer_norm.eps,
                      "feedforward_norm_eps": layer.feedforward_block.layer_norm.eps,
                      "gelu_approximate": layer.feedforward_block.activation.approximate,
                      "attention_dropout_training": layer.attention_block.mha.dropout,
                      "feedforward_dropout_training": layer.feedforward_block.dropout.p,
                      "dropout_effective_eval": 0.0,
                  } for layer in bundle["model"].layers],
                  "architecture_class_ast_matches": bundle["architecture_matches"]},
        "provenance": {"artifacts": artifacts,
                       "exporter_sha256": sha256(Path(__file__)),
                       "python": platform.python_version(), "pytorch": torch.__version__,
                       "device": "cpu", "dtype": "float32", "batch_size": 1,
                       "torch_num_threads": torch.get_num_threads(),
                       "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
                       "eval": True, "gradients_enabled": False, "output_subset": None,
                       "preprocessing": "Raw signed bounds; float32; zero non-input scalars; no scaling/sign flip.",
                       "test_solver": "pFBA according to maintained core experiment notes; no per-CSV solver sidecar",
                       "test_generation_seed": None, "test_pfba_fraction_of_optimum": None,
                       "generator_defaults": {"solver": "pfba", "pfba_fraction_of_optimum": 0.999,
                                              "training_seed": 42, "recommended_test_seed": 9},
                       "repository_sbml_only_ids": bundle["repository_sbml_only_ids"],
                       "checkpoint_only_ids_vs_repository_sbml": bundle["checkpoint_only_ids_vs_repository_sbml"],
                       "selection": {"scan_prefix_rows": scan_rows if rows is None else None,
                                     "explicit_rows": rows, "purpose": "illustration, not performance evaluation"}},
        "reactions": [{"index": i, "token": token, "reaction_id": token.removesuffix("_flux"),
                       "name": bundle["reaction_names"][token.removesuffix("_flux")],
                       "input_index": bundle["indices"].index(i) if i in bundle["indices"] else None}
                      for i, token in enumerate(bundle["outputs"])],
        "inputs": [{"input_index": j, "exchange_id": name, "token_index": bundle["indices"][j]}
                   for j, name in enumerate(bundle["inputs"])],
        "parameters": {},
        "first_layer_static": {name: traces[0]["layers"][0]["operations"][name] for name in static_names},
        "first_layer_independence": independence.report(),
    }
    for name, tensor in bundle["model"].state_dict().items():
        axes = ["parameter_axis_" + str(i) for i in range(tensor.ndim)]
        if name == "input_embedding.weight":
            axes = ["reaction", "embedding"]
        elif name == "input_token_indices":
            axes = ["input"]
        elif name.endswith("head_scores"):
            axes = ["head"]
        common["parameters"][name] = tensor_json(tensor, axes)
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "manifest", "common": "common.json.gz",
                "examples": [],
                "verification": {"passed": True, "atol": ATOL, "rtol": RTOL,
                                 "first_layer_attention_bitwise_independent": True}}
    for (row_index, record, reason), trace in zip(samples, traces):
        trace.update({"schema_version": SCHEMA_VERSION, "kind": "sample", "common": "common.json.gz",
                      "sample": {"csv_data_row_zero_based": row_index, "csv_line_one_based": row_index + 2,
                                 "selection_reason": reason, "source_record": record},
                      "initial_x": reference("common.json.gz#/parameters/input_embedding.weight"),
                      "final_predictions": reference(f"#/layers/{len(trace['layers'])-1}/operations/output_c")})
        for layer in trace["layers"]:
            i, operations = layer["index"], layer["operations"]
            layer["input_x"] = reference("#/initial_x" if i == 0 else f"#/layers/{i-1}/operations/output_x")
            layer["input_c"] = reference("#/initial_c" if i == 0 else f"#/layers/{i-1}/operations/output_c")
            operations["dropout"] = reference(f"#/layers/{i}/operations/gelu")
            operations["head_weights"] = reference(f"common.json.gz#/head_weights/{i}")
            if i == 0:
                for name in static_names:
                    operations[name] = reference(f"common.json.gz#/first_layer_static/{name}")
        filename = f"sample-{row_index:06d}.json.gz"
        write_json(output / filename, trace)
        checks = trace["verification"]["checks"]
        manifest["examples"].append({"file": filename, "sha256": sha256(output / filename),
                                     "row": row_index, "selection_reason": reason,
                                     "simulated_biomass": float(record["Biomass_Ecoli_core_flux"]),
                                     "check_count": len(checks),
                                     "max_absolute_error": max(check["max_absolute_error"] for check in checks),
                                     "max_relative_error": max(check["max_relative_error"] for check in checks),
                                     "max_tolerance_ratio": max(check["max_tolerance_ratio"] for check in checks)})
    # Head-combination weights are learned parameters, shared across conditions.
    common["head_weights"] = [tensor_json(layer.attention_block.head_scores.detach().softmax(0), ["head"])
                              for layer in bundle["model"].layers]
    write_json(output / "common.json.gz", common)
    manifest["common_sha256"] = sha256(output / "common.json.gz")
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2].parent / "metabolic-NN")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "examples")
    parser.add_argument("--rows", type=int, nargs="+", help="Explicit zero-based CSV data rows")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--scan-rows", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = export(load_bundle(args.repo), args.output, args.rows, args.count, args.scan_rows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
