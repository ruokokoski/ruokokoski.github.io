# Offline FluxFormer calculation traces

This directory implements the calculation-trace stage only. It does not change
the website, train a model, or implement browser inference. Jekyll ignores this
underscore-prefixed tools directory. Nothing here is wired into navigation.

## Run

From the website repository root, use the existing metabolic-NN environment:

```sh
../metabolic-NN/.venv/bin/python _tools/fluxformer_traces/export_traces.py
../metabolic-NN/.venv/bin/python -m unittest discover -s _tools/fluxformer_traces -p 'test_*.py' -v
```

The exporter defaults to the sibling `../metabolic-NN` repository and writes
`examples/`. It accepts portable explicit paths and sample selection:

```sh
../metabolic-NN/.venv/bin/python _tools/fluxformer_traces/export_traces.py --repo ../metabolic-NN --output /tmp/fluxformer-traces --rows 870 722 962 63
```

Without `--rows`, it scans the first 1,024 real test records, selects minimum
and maximum simulated biomass, then selects conditions farthest from those
already chosen in range-scaled nutrient bounds plus biomass. The default is
four conditions; `--count` supports three to five and `--scan-rows` changes the
prefix. Selection is deterministic and deliberately illustrative, not a
performance evaluation. Existing named output files are overwritten; use a
different `--output` directory to preserve a previous export.

Runtime dependencies are PyTorch and the standard library. Tests also use the
trainer's existing NumPy and pandas dependencies to execute its original data
loader. If installed, COBRA's bundled textbook SBML supplies exact reaction
names; otherwise the repository SBML supplies only exact-ID name matches, with
unmatched names left null. No packages are installed and no network is needed.
The trusted local checkpoint includes Python/NumPy RNG objects, so loading it
uses `torch.load(..., weights_only=False)`. Do not substitute an untrusted
pickle checkpoint.

## Artifact contract

The exporter fixes the target model to
`ecoli_core_10M_signed_pfba_d128_h8_l3_ff640` and loads its full
`*_checkpoint.pth`, including vocabulary, mapping, and configuration:

- 95 reaction tokens in checkpoint `data_info.output_cols` order;
- 20 signed exchange constraints mapped by exact `exchange_id + '_flux'`;
- three transformer layers, embedding width 128, eight heads of width 16,
  feed-forward width 640, and training dropout 0.02;
- float32 weights and computation on CPU, batch size one;
- best model epoch 108, ordinary Huber training;
- training CSV `2026-10-13_ecoli_core_training_data_10000000_samples.csv`;
- test CSV `2026-10-13_ecoli_core_test_data_100000_samples.csv`.

Both CSV headers must match the checkpoint exactly. Model loading is strict,
and checkpoint config, data metadata, and the saved input-index buffer must
agree. All four model-class ASTs in `flux_transformer.py` must match the
trainer's duplicate definitions. Tests also instantiate those original trainer
classes and compare their forward results with the canonical model.

Training preprocessing copies signed input values into matching scalar slots,
casts to float32, fills missing/NaN inputs with zero, and leaves other slots
zero. There is no input normalization or sign reversal. Constraints describe
uptake bounds, not realized exchange fluxes. Targets remain raw signed fluxes.

Every forward call uses `model.eval()`, disabled gradients, and
`output_subset=None`. All 95 tokens participate in all three layers. The scalar
after each layer is retained, with only the last marked as a final prediction.
The model also updates input-token scalars; it does not clamp them back to the
original nutrient bounds.

## What is verified

Read-only hooks capture the unchanged model's actual outputs. Separate explicit
calculations verify packed Q/K/V projections, scaled logits, multi-head mixing,
scalar diffusion, head-score softmax, both residual paths, normalization, and
the entire feed-forward block. References include:

- the original `nn.MultiheadAttention` outputs and per-head matrices;
- PyTorch's packed-projection helper and public functional MHA;
- scaled-dot-product attention, including identity-valued V to expose weights;
- float64 contraction/reductions for logits and LayerNorm, and the float64
  erf definition of exact GELU, rounded back to float32 for comparison;
- a separate end-to-end reconstruction which never calls a model block or MHA;
- fresh complete `model.forward()` calls and the original trainer's data loader.

Tests read the serialized JSON back and compare all operations with a fresh
original-model execution. Deliberately altered attention and final-output
entries must fail. The JSON parameters round-trip to the original state dict
bit for bit. The exporter stops on any failed numerical check.

The acceptance rule is elementwise
`abs(actual - reference) <= 2e-5 + 2e-5 * abs(reference)`. Reports retain maximum
absolute error, maximum relative error with denominator floor `1e-7`, and the
maximum ratio to that element's allowed tolerance. The relative diagnostic can
be large near zero and is not an independent rejection criterion. Tolerances
are fixed constants; they are not adjusted during generation.

See [VERIFICATION.md](VERIFICATION.md) for the measured results and limits, and
[SCHEMA.md](SCHEMA.md) for tensor layouts and access examples.

## Files

- `export_traces.py`: artifact validation, capture, independent reconstruction,
  verification, sample selection, and export.
- `test_traces.py`: integration and numerical regression tests using real
  artifacts; missing required artifacts fail rather than skip.
- `examples/manifest.json`: sample list, integrity hashes, and error summaries.
- `examples/common.json.gz`: metadata, all learned parameters, reaction mapping,
  shared first-layer computations, and condition-independence checks.
- `examples/sample-*.json.gz`: four full calculation traces, inputs, targets,
  raw CSV records, and individual numerical verification results.

The large JSON documents are gzip-compressed, with deterministic gzip headers.
They remain ordinary JSON after decompression. Python can inspect them without
any extra dependency:

```python
import gzip
import json
from pathlib import Path

root = Path('_tools/fluxformer_traces/examples')
manifest = json.loads((root / 'manifest.json').read_text())
with gzip.open(root / manifest['examples'][0]['file'], 'rt') as stream:
    trace = json.load(stream)
print(trace['sample'])
```

Keep the manifest, common file, and sample files together. A future static
frontend will need to decompress the `.json.gz` payloads explicitly unless its
host supplies the correct HTTP content encoding. No frontend behavior is
implemented in this stage.
