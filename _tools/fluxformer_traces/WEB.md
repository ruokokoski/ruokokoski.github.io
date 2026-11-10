# Static FluxFormer walkthrough

The page at `/fluxformer/` uses precomputed, verified traces. It has no backend,
production npm dependencies, browser model runtime, or editable nutrient bounds.
All nutrient conditions come from the four real simulated samples checked by
the original trace pipeline. Existing model source, checkpoints, and original
trace files are unchanged.

## Regenerate and preview

From the website root:

```sh
python3 _tools/fluxformer_traces/prepare_web_data.py
bundle3.2 exec jekyll build --strict_front_matter
bundle3.2 exec jekyll serve --host 127.0.0.1
```

Open `/fluxformer/` on the local preview. Default and prefixed Jekyll base URLs
use `relative_url` in page assets and the manifest URL. Data filenames resolve
against the manifest, not against the current page URL. The original
`export_traces.py` command remains the way to regenerate source traces from
the actual checkpoint and test CSV; packaging the saved traces needs only the
Python standard library.

The six steps have stable fragment links: `#initialize`, `#attention`, `#mix`,
`#feedforward`, `#layers`, and `#predictions`. Browser Back/Forward restores
steps. Condition, reaction, layer, and head selections stay in page memory;
they are not encoded in shareable fragment links.

## Data contract

`prepare_web_data.py` verifies the source manifest's common-file and sample
hashes and the numerical pass reports before packaging. It does not recompute
the model, normalize attention subsets, change precision, or round tensor data.

`data/fluxformer/manifest.json` includes:

- The exact model configuration, reaction IDs, names, indices, and input map.
- All 95 initial embeddings and all three layers' learned scalar head weights.
- Each feed-forward second-linear map's scalar-output weight row and bias,
  for inspecting a hidden feature's contribution.
- Source-model/data provenance, source trace hashes, verification metadata, and
  filenames/hashes for all four conditions.

Each `sample-NNNNNN.json` includes the 20 original signed bounds, all 95 initial
scalar values, three complete 95-scalar layer states, all 95 simulated targets,
and the source numerical verification report. It is sufficient for initialization,
the layer-state chart, and final prediction comparisons.

Each `sample-NNNNNN-layer-I.json.gz` includes the layer's full input x/c and
every named operation from the original trace, with reference/tensor wrappers
resolved into ordinary nested arrays. Attention retains its full `[8,95,95]`
shape. Q/K/V are `[8,95,16]`; feed-forward hidden values are `[95,640]`.
No reaction or feature is removed. See [SCHEMA.md](SCHEMA.md) for operation
semantics. The public layer document records the sample row and layer index.

The initial manifest is approximately 290 KiB and each sample overview is
approximately 24 KiB, before HTTP compression. A layer download is approximately
4.3 MiB in gzip form; the whole public dataset is approximately 52 MiB. The
large layers load only for attention, mixing, or feed-forward steps. At most
two layer promises/decoded results are cached; older entries are evicted.

Layer downloads use native `DecompressionStream('gzip')`. The loader recognizes
both raw gzip bytes and responses already decompressed by the HTTP host. A
browser without the API can still inspect initialization, layer scalar states,
and final predictions; detailed steps show a clear explanation and retain
navigation. No external decompression library or CDN is used.

## Display and scientific interpretation

Initialization keeps six example tokens fixed and displays the selected reaction
in a separate inspection panel. Clicking an example or choosing a reaction
updates that panel and the reaction followed through later steps. Other steps
show five attention sources or one selected reaction's calculation. All 95 reactions remain accessible through
native reaction/source selectors; final results have search and pagination
with six rows per page. The selected sample, head, and layer apply consistently
to the visible numerical values.

Initial c entries are uptake lower bounds, not realized fluxes. Non-input
entries start at zero. Attention's last axis enumerates source reactions;
its row is the receiving/query token. The five-source chart retains original
weights rather than renormalizing them. It explicitly describes neural
information mixing, not biochemical or causal connections.
Step 2 explains all 95 query–key comparisons (including self-attention), the
16 features per head (128 ÷ 8), division by √16 = 4, and row-wise softmax
across all 95 scores. The displayed pair is labeled as one entry in the row;
the full pass produces eight 95 × 95 attention matrices.

The interface denotes the 95 × 128 embedding matrix by X. Each reaction vector
is one row, and X[i, j] selects reaction i's feature j using zero-based indices.
Original trace keys such as `initial_x` remain unchanged.
The six example tokens preview features 0–7. The selected panel displays all
128 features, in one row on desktop and two rows on narrow screens. Negative
features are blue and extend below zero; positive features are red and extend
above zero. Heights use the vector's maximum absolute feature as their scale.
Feature tooltips retain the exact index and recorded value.

The scalar path uses learned `softmax(head_scores)` to combine heads. The
embedding path concatenates heads and applies the output projection. Both add
their input through residual connections. The feed-forward block joins x/c,
normalizes 129 features, maps to 640 hidden features, applies exact GELU, maps
back to 129 features, adds the joined residual, and splits x/c again.

Layer 1 attention is condition-independent because its initial embeddings are
condition-independent; later layers can respond to nutrient context through
the feed-forward updates. Earlier scalar states are labeled intermediate.
Only the layer 3 scalar is labeled a final prediction. The targets are simulated
outputs, not experimental measurements; the model does not enforce metabolic
constraints at inference.

Numbers are rounded only when displayed; hovering a formatted scalar exposes
its complete recorded JSON value in the native title tooltip. Browsers display
the recorded logits, attention weights, intermediate outputs, and final
predictions. Simple arithmetic for explanatory dot-product terms, stable
softmax denominators, scalar contribution terms, and absolute errors uses
JavaScript doubles. It does not replace recorded float32 operation results or
produce new predictions. Softmax and scalar explanations are tested against
recorded outputs at the original float32 tolerances.

Historical generator settings remain qualified as in [VERIFICATION.md](VERIFICATION.md).
These selected examples demonstrate computation and are not a performance
benchmark or evidence of experimental accuracy.

## Tests

```sh
# Original model checks plus complete source-to-public tensor equality checks
../metabolic-NN/.venv/bin/python -m unittest discover -s _tools/fluxformer_traces -p 'test_*.py' -v

# Native JavaScript helpers, all four samples/three layers, gzip, cache/retry,
# float32 explanation consistency, formatting, and browser-fetch receiver
node --experimental-default-type=module _tools/fluxformer_traces/test_web.mjs

# Integration build
bundle3.2 exec jekyll build --strict_front_matter
```

The JavaScript command uses Node 20's module flag without introducing a root
package manifest. `test_web_data.py` compares every published layer operation,
all 95 tokens and targets, the initialization values, scalar-projection weights,
and all file hashes with the verified original traces.

Browser tests need Playwright as an optional development tool, and a local
preview. They are not shipped to the browser or required to build the site.
For example, with an existing Playwright installation:

```sh
FLUXFORMER_PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs FLUXFORMER_PREVIEW_URL=http://127.0.0.1:4175 node _tools/fluxformer_traces/browser_test.mjs
```

`browser_test.mjs` exercises Chromium and WebKit: numerical displays, every
walkthrough step, all contextual selectors, source selection with keyboard
focus, feature bounds, search/pagination, history, dark/light themes, and page
overflow at widths 320, 390, 768, 1024, and 1440. Separate cases cover manifest
and layer failures/retries, a delayed stale sample response, missing gzip
support, and disabled JavaScript. Screenshots default to
`/tmp/fluxformer-browser-screenshots/`.

## Verified results (2026-11-10)

- All 12 Python tests pass: eight original PyTorch trace checks and four
  complete source-to-published-data checks.
- All five JavaScript tests pass, covering four samples and three layers,
  numerical explanation consistency, caching, retry, and loader compatibility.
- All 215 browser assertions pass in Chromium and WebKit, including every
  walkthrough stage at widths 320, 390, 768, 1024, and 1440px, both themes,
  keyboard focus, history, error recovery, and delayed-response protection.
  Selection checks verify that the six examples remain fixed while the
  separate inspection panel follows the chosen reaction and its exact data.
  Feature checks verify all 128 values and indices, blue/red sign colors in
  both themes, and that the complete vector fits at every tested width.
- Normal and `/preview`-prefixed Jekyll builds pass. The prefixed build includes
  trace assets, resolves the prefixed page/manifest/script URLs, and excludes
  the offline tools directory.
- Published tensor values are exactly equal to the source traces; packaging
  introduces no numerical error. Recorded predictions still match the
  original PyTorch forward as documented in the source verification report.
- Desktop, mobile, light-mode, feed-forward-detail, and results screenshots
  were reviewed. No page-level overflow remains at the tested widths.
- JavaScript syntax checks and `git diff --check` pass.

These are local build and browser results. No commit, push, or publication was
performed. Remaining limits are the four fixed illustrative conditions,
approximately 4.3 MiB per lazily loaded layer, the modern-browser requirement
for detailed gzip traces, and the original dataset-provenance qualifications.
