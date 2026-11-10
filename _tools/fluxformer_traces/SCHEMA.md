# FluxFormer trace schema 1.0.0

All documents carry `schema_version` and `kind`. Indices are zero-based. The
batch axis is omitted from exported tensors because every example has batch
size one. Dimensions are `S=95` reactions, `D=128` embedding features, `H=8`
heads, `d=16` head features, `J=D+1=129` joined features, and `F=640` hidden
features. The checkpoint's output vocabulary fixes the reaction axis everywhere.

## Tensor values and references

Every tensor is either an object containing `dtype`, `shape`, `axes`, and
`data` (nested row-major JSON arrays), or a reference object:

```json
{"$ref":"common.json.gz#/first_layer_static/attention"}
```

References use JSON Pointer paths following `#`. An empty filename addresses
the current sample document; `common.json.gz` addresses the common document.
References can chain, for example layer-zero `input_x` → sample `initial_x` →
common embedding parameters. Array path components are decimal indices.
Parameter names containing dots are literal dictionary keys, not nested paths.
All floating tensors are float32, and the input-index buffer is int64. JSON
numbers are written with round-trip precision; consumers must not round them
before calculation. No NaN or infinity is emitted.

## Manifest

`manifest.json` contains:

| Field | Meaning |
| --- | --- |
| `common`, `common_sha256` | Shared document filename and compressed-file hash |
| `examples[]` | File, hash, CSV data-row index, selection reason, simulated biomass, check count, maximum errors |
| `verification` | Acceptance tolerances and first-layer independence result |

## Common document

`common.json.gz` contains:

| Field | Meaning |
| --- | --- |
| `model` | Exact model name, checkpoint config, epoch, training-data metadata, per-layer epsilon/GELU/dropout settings, architecture AST checks |
| `provenance` | Source/checkpoint/data/name-artifact hashes, exporter hash, runtime, precision, inference settings, selection, preprocessing, qualified solver provenance |
| `reactions[S]` | `index`, original `token`, exact `reaction_id`, name or null, input index or null |
| `inputs[20]` | Input-column index, exchange ID, and matching reaction-token index |
| `parameters` | Every original model state-dict entry, keyed by its exact parameter/buffer name |
| `head_weights[3]` | Learned `softmax(head_scores)` for each layer, shape `[H]` |
| `first_layer_static` | The condition-independent first-layer embedding calculations listed below |
| `first_layer_independence` | Numerical comparisons across all selected real conditions; bitwise equality is additionally enforced |

`parameters['input_embedding.weight']` is the initial `[S,D]` embedding tensor.
For layer `i`, the packed projection weight key is
`layers.i.attention_block.mha.in_proj_weight`, shape `[3D,D]`. Its row blocks
`[0:D]`, `[D:2D]`, and `[2D:3D]` are Q, K, and V respectively; biases follow
the same partition. Linear weights use PyTorch's `[out_feature,in_feature]`
convention, so `linear(u,W,b) = u @ W.T + b`.

First-layer `attention_norm`, `query`, `key`, `value`, `scaled_logits`,
`attention`, `weighted_values`, `combined_heads`, `output_projection`, and
`embedding_residual` are stored once under `first_layer_static`. They depend
only on learned embeddings and weights. The scalar operations and feed-forward
results remain sample-specific. Later-layer attention remains sample-specific.

## Sample document

`sample-NNNNNN.json.gz` contains:

| Field | Meaning |
| --- | --- |
| `common` | Shared metadata filename |
| `sample` | Exact zero-based CSV data row, one-based CSV line, selection reason, original string-valued CSV record |
| `constraints` | All 20 signed nutrient bounds, `[20]`, in `common.inputs` order |
| `initial_x` | Reference to the learned embedding matrix, `[S,D]` |
| `initial_c` | Injected bounds and zero non-input entries, `[S,1]` |
| `targets` | All simulated target fluxes, `[S]` |
| `layers[3]` | Index, state kind, references to input states, and named operation tensors |
| `final_predictions` | Reference to the last layer's `output_c`, `[S,1]` |
| `verification` | 85 checks, errors, tolerances, and pass/fail results |

`layers[i].state_kind` is `intermediate` for layers 0 and 1 and `final` for
layer 2. Layer indices do not imply that an early scalar state is already a
validated standalone predictor.

## Layer operations

The following names live under `layers[i].operations`. Let `X,C` be that
layer's input states, `A` its attention, and `alpha=softmax(head_scores)`.
LayerNorm normalizes each reaction's feature axis, with learned affine terms
and the exported epsilon. There is no positional encoding, attention mask,
or nutrient projection into Q/K in this model.

| Operation | Shape | Calculation |
| --- | --- | --- |
| `attention_norm` | `[S,D]` | LayerNorm(X) |
| `query`, `key`, `value` | `[H,S,d]` | Packed linear projection of normalized X, partitioned and reshaped into heads |
| `scaled_logits` | `[H,S,S]` | `(Q * sqrt(1/d)) @ K.T`, scaling before multiplication as in PyTorch functional MHA |
| `attention` | `[H,S,S]` | Native MHA's softmax weights; last axis sums to approximately one |
| `weighted_values` | `[H,S,d]` | `A @ V` |
| `combined_heads` | `[S,D]` | Transpose heads after reaction axis, concatenate head features |
| `output_projection` | `[S,D]` | Native MHA embedding output, verified against linear projection of combined heads |
| `embedding_residual` | `[S,D]` | `X + output_projection` |
| `scalar_heads` | `[H,S,1]` | `A @ C` |
| `head_weights` | `[H]` | Reference to common alpha for this layer |
| `scalar_combined` | `[S,1]` | Sum of `alpha[h] * scalar_heads[h]` over heads |
| `scalar_residual` | `[S,1]` | `C + scalar_combined` |
| `concatenated` | `[S,J]` | Join embedding_residual and scalar_residual on the feature axis |
| `feedforward_norm` | `[S,J]` | LayerNorm(concatenated), including the scalar feature |
| `linear1` | `[S,F]` | First feed-forward affine map |
| `gelu` | `[S,F]` | Exact GELU (`approximate='none'`) |
| `dropout` | `[S,F]` | Reference to GELU, because dropout is identity in eval mode |
| `linear2` | `[S,J]` | Second feed-forward affine map |
| `feedforward_residual` | `[S,J]` | `concatenated + linear2` |
| `output_x` | `[S,D]` | First D features of feedforward_residual |
| `output_c` | `[S,1]` | Last feature of feedforward_residual |

`attention_norm`, native `attention`, `output_projection`, both attention
residual outputs, feed-forward module outputs, and final X/C are captured
directly from the original model. Packed projections, logits, headwise products,
head concatenation, scalar mixing, and concatenated/residual tensors are
reconstructed and independently verified. Small differences between explicit
matmul, fused/native kernels, SDPA, and float64 reference calculations are
reported rather than rounded away. No approximate replacement is used for a
failed operation; export aborts on failure.

The MHA output combines embedding heads by concatenation followed by its
learned output projection. The learned alpha weights combine only the scalar
heads. These are separate operations.

## Retrieving a reaction, layer, head, or operation

After loading/decompressing both JSON documents, resolve `$ref` before reading
a tensor's `data`. A small standard-library resolver is:

```python
def tensor(document, value, common):
    while '$ref' in value:
        filename, pointer = value['$ref'].split('#', 1)
        value = common if filename == 'common.json.gz' else document
        for key in pointer.lstrip('/').split('/'):
            value = value[int(key)] if isinstance(value, list) else value[key]
    return value['data']

reaction_index = next(r['index'] for r in common['reactions']
                      if r['reaction_id'] == 'Biomass_Ecoli_core')
layer_index, head_index = 0, 2
operations = trace['layers'][layer_index]['operations']
attention = tensor(trace, operations['attention'], common)
row = attention[head_index][reaction_index]  # weights over ALL 95 source reactions
biomass_after_layer = tensor(trace, operations['output_c'], common)[reaction_index][0]
simulated_biomass = tensor(trace, trace['targets'], common)[reaction_index]
```

The attention row's first reaction axis is the **receiving/query reaction**;
the last axis enumerates the **source reactions** whose values are mixed into
it. Selecting six to eight visible reactions must only select a view of these
arrays. Never renormalize a displayed submatrix or run attention on the visible
subset. Attention mixing is a neural calculation, not a biochemical reaction
or proof of causal metabolic influence.

## Numerical report format

Each check includes `operation`, full internal `shape` (including batch where
applicable), `max_absolute_error`, `max_relative_error`, `max_tolerance_ratio`,
and `passed`. The ratio is the maximum of
`abs(error)/(atol + rtol*abs(reference))`; it must be at most one. Maximum
relative error uses `max(abs(reference),1e-7)` as the denominator. Maxima can
occur at different elements and must not be combined as though they described
a single element. The report checks high-precision references rounded to
float32; exported tensors preserve actual float32 inference precision.
