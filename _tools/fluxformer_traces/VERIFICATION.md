# Numerical verification report

Verified on 2026-11-10 using the existing metabolic-NN environment: Python
3.12.3, PyTorch 2.7.1+cu126, CPU float32, one CPU thread, batch size one,
eval mode, inference mode, and full-vocabulary inference. CUDA was not used.

## Model and data

The selected real checkpoint is
`models/ecoli_core_10M_signed_pfba_d128_h8_l3_ff640/ecoli_core_10M_signed_pfba_d128_h8_l3_ff640_checkpoint.pth`,
SHA-256 `e0330447fcc293bb4d0908f804ba90713b83e1c70754bd806eeec430f73fe200`.
It contains best model epoch 108, with validation loss
`0.0040008755319565536`, and config d128/h8/l3/ff640/dropout0.02.

The independent test CSV is
`data/2026-10-13_ecoli_core_test_data_100000_samples.csv`, SHA-256
`2c58775788e0b92deafbc0d57bdcc4413707b6f2d73adde99b49d0d585f8fc68`.
Both training and test CSV headers match the checkpoint's ordered 20 inputs
and 95 output tokens. The checkpoint's saved input-index buffer, config, and
data metadata all agree on this mapping:

```text
[27, 25, 32, 37, 19, 21, 38, 26, 33, 23, 20, 24, 28, 29, 22, 30, 31, 34, 35, 36]
```

All four canonical model-class ASTs match the trainer's duplicate definitions.
Instantiating the original trainer classes and strictly loading the same state
dict gives bitwise-identical full forward predictions on the selected samples.
All model parameter and buffer tensors survive JSON serialization bit for bit.

## Numerical results

The exporter performs 85 checks per sample: **340 checks across four real
conditions**, plus **30 comparisons** of the condition-independent first-layer
embedding path. All pass the fixed elementwise tolerance
`abs(error) <= 2e-5 + 2e-5*abs(reference)`.

| Zero-based CSV data row | Simulated biomass | Maximum absolute error across all checks | Maximum relative error | Maximum fraction of allowed tolerance |
| --- | --- | --- | --- | --- |
| 870 | 0.0015953884 | 7.6293945e-6 | 0.8277399 | 0.0734829 |
| 722 | 2.2510213428 | 7.6293945e-6 | 0.8276588 | 0.0513566 |
| 962 | 1.1921629924 | 7.6293945e-6 | 0.8277329 | 0.1140614 |
| 63 | 0.3336676582 | 5.7220459e-6 | 0.8280340 | 0.0506213 |

Maximum absolute and relative errors are separate diagnostics and can come
from different checks and different elements. The largest relative errors
occur in GELU's near-zero negative tail when comparing PyTorch float32 GELU
with its independent float64 erf definition rounded back to float32. The
maximum absolute error in those GELU comparisons is at most `4.7683716e-7`.
The relative denominator is floored at `1e-7`; consequently even a tiny absolute
tail difference can produce a large relative diagnostic. These comparisons
use less than 1.3% of the allowed elementwise tolerance. They do not replace
the original model's GELU outputs, which are captured directly in the traces.

Other nonzero errors arise in float64 logit references, independent SDPA,
functional MHA, and scalar contractions with different reduction orders. The
largest tolerance ratio anywhere is `0.1140614`, comfortably below one. No
tolerance was increased to accommodate a failure.

For all four examples:

- Exported scalar states after **every layer** versus a fresh original
  `model.forward()`: maximum absolute and relative errors **0**.
- Separate end-to-end reconstruction of all three layers, without invoking a
  model block or MHA, versus original final flux predictions: maximum absolute
  and relative errors **0** on this runtime.
- First-layer attention and the rest of its embedding path across the selected
  nutrient conditions: **bitwise identical**.
- First-layer scalar outputs differ, and second-layer attention differs between
  the low- and high-biomass examples. This follows the architecture: initial
  embeddings determine first-layer attention, while the feed-forward block
  injects scalar context into embeddings used by later layers.

Each sample's `verification.checks` contains the complete operation-specific
errors. `examples/manifest.json` contains the summary and compressed-file
integrity hashes.

## Automated tests

The documented unittest command passes **8 tests**:

1. Original trainer preprocessing and exact input-token mapping, including
   missing/NaN inputs and non-input zero initialization.
2. Original trainer-class versus canonical-model full forward equality.
3. Every serialized operation versus independent fresh original-model outputs.
4. Numerical report coverage of every layer and end-to-end reconstruction.
5. Detection of deliberately corrupted attention and final predictions.
6. Explicit rejection of infinite inputs, incompatible positive bounds, and
   out-of-range sample rows.
7. First-layer attention independence and changing later states.
8. File-hash integrity and bitwise JSON round-trip of all trained parameters.

`git diff --check` also passes. Existing uncommitted website changes were
preserved. No model, training script, website template, stylesheet, JavaScript,
or navigation file was changed for this task.

## Provenance limits

The repository's `models/e_coli_core.xml` has `BIOMASS_Ecoli_core_w_GAM` and
`FORt`, whereas the checkpoint uses `Biomass_Ecoli_core` and `FORti`. Those IDs
were not aliased or changed. Exact names for all 95 checkpoint reactions were
obtained from the locally installed COBRA textbook model, whose reaction set
matches exactly; its compressed SBML hash is recorded in the common document.
Matching IDs and obtaining names do not establish that two metabolic models
have identical stoichiometry. The source-model file used to generate this CSV
was not saved alongside it with a model hash.

The maintained experiment notes identify the selected test data as pFBA.
There is no per-CSV sidecar proving the historical generation command, random
seed, exact solver version, or fraction of optimum. Those sample-provenance
fields remain unknown/null. The current generator defaults (pFBA 0.999,
training seed 42, recommended independent-test seed 9) are recorded separately
and are not asserted as measured provenance of this CSV.

The training CSV's header was checked; its multi-gigabyte contents were not
hashed or audited. Numerical verification concerns faithful model computation,
not predictive accuracy, physiological validity, or stoichiometric feasibility.
The four deliberately selected examples do not estimate generalization.
Results above apply to the recorded CPU/PyTorch runtime; other kernels or
devices can differ within float32 precision and must be verified again.

The gzip JSON examples retain all 95 reactions and all three layers. Shared
parameters and first-layer static computations are stored once. The files
are intentionally offline artifacts; frontend implementation stops at this
stage.
