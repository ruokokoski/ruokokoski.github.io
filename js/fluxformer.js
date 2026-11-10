import { TraceStore, steps, format, topSources, softmaxEvidence, scalarEvidence, trajectory, attentionPairEvidence } from "./fluxformer-data.js";

const root = document.getElementById("fluxformer");
if (root) start();

function start() {
  const $ = id => document.getElementById(`ff-${id}`);
  const escape = value => String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const exact = value => escape(String(value));
  const number = value => `<span class="ff-number" title="${exact(value)}">${format(value)}</span>`;
  const percent = value => `${(value * 100).toFixed(2)}%`;
  const state = { sample: 722, reaction: 12, layer: 0, head: 0, source: null, feature: 0, hidden: 0,
    nutrient: 0, pairFeature: 0, channel: "scalar", step: Math.max(0, steps.indexOf(location.hash.slice(1))), search: "", page: 0 };
  let store, meta, overview, layer, revision = 0;
  const titles = ["Start with a nutrient condition", "Compare one query with all 95 keys", "Use attention to update both channels",
    "Let the two channels interact", "Refine the states across three layers", "Read the final flux predictions"];
  const descriptions = [
    "The 95 reaction vectors form an embedding matrix X with 128 features per row. Each reaction also has a scalar c. Nutrient bounds initialize c at their matching tokens; the other scalars start at zero.",
    "Learned projections turn normalized embeddings into queries Q and keys K. The selected receiving reaction’s query is compared with every token’s key, including its own.",
    "The same attention matrices mix embedding values and scalar values. Each path then adds its input back through a residual connection.",
    "The feed-forward network joins each row of X with its scalar c, transforms them together, and adds a residual. This is where nutrient context can change the reaction embeddings.",
    "The updated X and c become the next layer’s inputs. An early scalar is an intermediate state; only the last layer supplies the final prediction.",
    "The final scalar for each reaction is its predicted flux. Compare it with the matching simulated target under exactly the same nutrient bounds.",
  ];
  const shortNames = { EX_glc__D_e: "Glucose", EX_fru_e: "Fructose", EX_o2_e: "Oxygen", EX_pi_e: "Phosphate",
    EX_nh4_e: "Ammonium", EX_co2_e: "CO₂ exchange", Biomass_Ecoli_core: "Growth (biomass)", PFK: "PFK", ATPS4r: "ATP synthase" };
  const reaction = index => meta.reactions[index];
  const name = index => shortNames[reaction(index).reaction_id] || reaction(index).reaction_id;
  const option = (index, selected) => `<option value="${index}"${index === selected ? " selected" : ""}>${escape(reaction(index).reaction_id)} — ${escape(reaction(index).name || "Reaction")}</option>`;
  const reactionOptions = selected => meta.reactions.map((_, index) => option(index, selected)).join("");
  const card = (label, value, type = "") => `<div class="ff-value ${type}"><span>${label}</span><strong>${number(value)}</strong></div>`;
  const detail = (key, title, content) => `<details class="ff-detail" data-key="${key}"><summary>${title}</summary><div class="ff-detail-body">${content}</div></details>`;
  const equation = (text, caption) => `<div class="ff-equation">${text}</div>${caption ? `<p class="ff-caption">${caption}</p>` : ""}`;
  const rowTable = (headers, rows, caption = "") => `<div class="ff-table-wrap"><table>${caption ? `<caption>${caption}</caption>` : ""}<thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((cell, i) => i === 0 ? `<th scope="row">${cell}</th>` : `<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  const vectorLegend = '<span class="ff-vector-legend"><span class="negative">− Negative</span><span class="positive">+ Positive</span></span>';
  const valuesPreview = (values, count = 8) => {
    const full = count >= values.length;
    const maximum = Math.max(...values.map(Math.abs)) || 1;
    return `<div class="ff-vector${full ? " ff-vector-full" : ""}" role="img" aria-label="${full ? "All" : "First"} ${Math.min(count, values.length)}${full ? "" : ` of ${values.length}`} embedding features; blue is negative, red is positive; bar height indicates magnitude">${values.slice(0, count).map((value, i) => `<span class="ff-feature ${value < 0 ? "negative" : value > 0 ? "positive" : "zero"}" data-feature="${i}" data-value="${exact(value)}" title="Feature ${i}: ${exact(value)}" style="--level:${Math.abs(value) / maximum * 50}%"><i aria-hidden="true"></i></span>`).join("")}${full ? "" : "<b>…</b>"}</div>`;
  };
  const featureField = (id, label, value, maximum) => `<label class="ff-field ff-small-field" for="ff-${id}"><span>${label} (0–${maximum})</span><input id="ff-${id}" type="number" min="0" max="${maximum}" step="1" value="${value}"></label>`;
  const unitNote = () => state.reaction === 12 ? "Growth rate in h⁻¹." : "Reaction flux in mmol gDW⁻¹ h⁻¹; the sign follows the model’s reaction direction.";

  async function initialize() {
    $("retry").hidden = true;
    $("status").hidden = false;
    $("status").textContent = "Loading verified conditions…";
    store = new TraceStore(new URL(root.dataset.manifest, location.href));
    try {
      meta = await store.metadata();
      if (!meta.samples.some(sample => sample.row === state.sample)) state.sample = meta.samples[0].row;
      $("sample").innerHTML = meta.samples.map(sample => `<option value="${sample.row}">${escape(sample.label)} · row ${sample.row}</option>`).join("");
      $("sample").value = state.sample;
      $("reaction").innerHTML = reactionOptions(state.reaction);
      $("model-source").textContent = `Checkpoint: ${meta.model.name}, epoch ${meta.model.model_epoch}. Source checkpoint SHA-256: ${meta.provenance.artifacts.checkpoint.sha256}.`;
      $("app").hidden = false;
      $("status").hidden = true;
      await render();
    } catch (error) {
      $("status").textContent = error.message;
      $("retry").hidden = false;
    }
  }

  function initializeView() {
    const shown = ["EX_glc__D_e", "EX_o2_e", "EX_pi_e", "PFK", "ATPS4r", "Biomass_Ecoli_core"]
      .map(id => meta.reactions.find(r => r.reaction_id === id).index);
    const nutrient = meta.inputs[state.nutrient];
    const current = reaction(state.reaction);
    const tokens = shown.map(index => `<button class="ff-token${index === state.reaction ? " is-selected" : ""}" type="button" data-reaction="${index}" aria-pressed="${index === state.reaction}">
      <span>${escape(name(index))}</span><code>${escape(reaction(index).reaction_id)}</code>
      ${valuesPreview(meta.initial_x[index])}<span class="ff-channel-label ff-x">Row of X · 128 learned features</span>
      <strong class="ff-c">c = ${number(overview.initial_c[index])}</strong></button>`).join("");
    return `<div class="ff-visual-label"><span>Reaction embeddings <b class="ff-x">X</b></span><span>Nutrient context <b class="ff-c">c</b></span></div>
      <div class="ff-token-grid">${tokens}</div>
      ${vectorLegend}
      <p class="ff-caption">The first eight features are previewed for six fixed example tokens. Click an example or use “Inspect reaction” to choose any reaction. Its full embedding appears below.</p>
      <section class="ff-inspection" aria-labelledby="ff-inspection-title">
        <p class="ff-inspection-label">Selected reaction · followed through all six steps</p>
        <header><div><h3 id="ff-inspection-title">${escape(name(state.reaction))}</h3><code>${escape(current.reaction_id)}</code></div>
          <div class="ff-initial-scalar"><span>Initial scalar c</span><strong class="ff-c">${number(overview.initial_c[state.reaction])}</strong></div></header>
        <div class="ff-inspection-embedding"><span class="ff-x">Row ${state.reaction} of X · all 128 learned features</span>${vectorLegend}${valuesPreview(meta.initial_x[state.reaction], 128)}<span class="ff-vector-order">Features 0–127, ordered left to right, then top to bottom. Bar height shows magnitude relative to this vector’s largest feature.</span></div>
        <p>${current.input_index === null
        ? `This reaction has no supplied nutrient bound. Its scalar starts at ${number(0)}; its embedding is learned during training.`
        : `The signed bound ${number(overview.initial_c[state.reaction])} is copied into this token’s scalar. A negative bound allows uptake up to that magnitude; it is not the amount actually taken up.`}</p>
        ${detail("embedding", "Inspect one embedding feature", `<p>Each reaction starts with <strong>128 learned embedding features</strong>. These numbers are initialized randomly before training, then adjusted to help the model compute attention and predict fluxes. The walkthrough uses their trained values.</p>
          <p>The 95 reaction vectors form the <strong>95 × 128 embedding matrix X</strong>. <code>X[i, j]</code> means feature <code>j</code> of reaction <code>i</code>, counting from zero. Individual features have no predefined biological meaning.</p>
          ${featureField("feature", "Embedding feature", state.feature, 127)}
          ${equation(`X[${state.reaction}, ${state.feature}] = ${number(meta.initial_x[state.reaction][state.feature])}`, "The initial embedding depends on reaction identity, so it is the same under every nutrient condition.")}`)}
      </section>
      ${detail("nutrients", "Inspect a nutrient bound and its matching token", `
        <label class="ff-field" for="ff-nutrient"><span>Nutrient input</span><select id="ff-nutrient">${meta.inputs.map((input, i) => `<option value="${i}"${i === state.nutrient ? " selected" : ""}>${escape(input.exchange_id)}</option>`).join("")}</select></label>
        ${rowTable(["Recorded quantity", "Value"], [["Signed uptake lower bound", number(overview.constraints[state.nutrient])], ["Initial scalar at token " + nutrient.token_index, number(overview.initial_c[nutrient.token_index])], ["Simulation target flux", number(overview.targets[nutrient.token_index])]])}
        <p>Bound ${escape(nutrient.exchange_id)} maps to <code>${escape(reaction(nutrient.token_index).token)}</code>. The target is the simulator’s resulting flux, which can differ from the bound. Units: mmol gDW⁻¹ h⁻¹.</p>`)}`;
  }

  function attentionDiagram(r, j, h) {
    const pair = attentionPairEvidence(layer, r, j, h);
    const f = state.pairFeature;
    const queryY = 10 + (r + 0.5) / 95 * 130;
    const keyX = 10 + (j + 0.5) / 95 * 160;
    const cellX = 25 + (j + 0.5) / 95 * 130;
    const cellY = 10 + (r + 0.5) / 95 * 130;
    const vector = (values, label) => {
      const maximum = Math.max(...values.map(Math.abs)) || 1;
      return `<div class="ff-pair-vector"><span>${label}</span><div class="ff-pair-cells" role="img" aria-label="${escape(label.replace(/<[^>]*>/g, ""))}: all 16 recorded features">${values.map((value, i) => `<span class="ff-pair-cell ${value < 0 ? "negative" : value > 0 ? "positive" : "zero"}${i === f ? " is-active" : ""}" data-feature="${i}" data-value="${exact(value)}" title="Feature ${i}: ${exact(value)}" style="--strength:${15 + Math.abs(value) / maximum * 75}%"></span>`).join("")}</div></div>`;
    };
    return `<figure class="ff-matrix-walkthrough" data-query="${r}" data-key="${j}" data-head="${h}" data-layer="${state.layer}">
      <figcaption><strong>One head: a row meets a column</strong><span>Highlights follow the receiving reaction and selected source. Matrix shapes are schematic; indices identify their actual positions.</span></figcaption>
      <label class="ff-field ff-matrix-source" for="ff-source"><span>Source reaction (key)</span><select id="ff-source" aria-describedby="ff-source-help">${reactionOptions(j)}</select></label>
      <p id="ff-source-help" class="ff-caption">The receiving reaction selected above supplies Q; this source supplies K. Selecting another source changes the comparison shown. All 95 sources still participate.</p>
      <div class="ff-matrix-product">
        <div class="ff-matrix-card"><h3>Queries Q</h3><span>95 × 16</span><svg viewBox="0 0 180 150" role="img" aria-label="Query matrix: row ${r} highlighted"><rect class="ff-matrix-body" x="75" y="10" width="22" height="130"/><path class="ff-matrix-guide" d="M75 36H97 M75 62H97 M75 88H97 M75 114H97"/><rect class="ff-query-mark" x="75" y="${queryY - 1.5}" width="22" height="3"/><path class="ff-query-line" d="M61 ${queryY}H111"/></svg><p>Row <strong>${r}</strong><small>Receiving token</small></p></div>
        <span class="ff-matrix-operator" aria-label="multiplied by">×</span>
        <div class="ff-matrix-card"><h3>Keys Kᵀ</h3><span>16 × 95</span><svg viewBox="0 0 180 150" role="img" aria-label="Transposed key matrix: column ${j} highlighted"><rect class="ff-matrix-body" x="10" y="62" width="160" height="27"/><path class="ff-matrix-guide" d="M42 62V89 M74 62V89 M106 62V89 M138 62V89"/><rect class="ff-key-mark" x="${keyX - 1.5}" y="62" width="3" height="27"/><path class="ff-key-line" d="M${keyX} 49V102"/></svg><p>Column <strong>${j}</strong><small>Source token</small></p></div>
        <span class="ff-matrix-operator" aria-label="equals">=</span>
        <div class="ff-matrix-card"><h3>Scores QKᵀ</h3><span>95 × 95</span><svg viewBox="0 0 180 150" role="img" aria-label="Score matrix: receiving row ${r}, source column ${j}, selected cell ${r}, ${j}"><rect class="ff-matrix-body" x="25" y="10" width="130" height="130"/><path class="ff-matrix-guide" d="M25 36H155 M25 62H155 M25 88H155 M25 114H155 M51 10V140 M77 10V140 M103 10V140 M129 10V140"/><rect class="ff-score-row" x="25" y="${cellY - 2}" width="130" height="4"/><path class="ff-key-line" d="M${cellX} 10V140"/><circle class="ff-score-cell" cx="${cellX}" cy="${cellY}" r="4"/></svg><p>Cell <strong>[${r}, ${j}]</strong><small>One of 95 in this row</small></p></div>
      </div>
      <div class="ff-pair-calculation">
        <h3>Match the 16 features</h3><p class="ff-caption">The selected Q row and K column are aligned horizontally here. Blue is negative, red is positive; outlines mark feature ${f}.</p>
        ${vector(pair.query, `Q<sub>${r}</sub>`)}${vector(pair.key, `K<sub>${j}</sub>`)}
        ${featureField("pair-feature", "Head feature to inspect", f, 15)}
        <div class="ff-feature-product" data-feature="${f}"><span>Feature ${f} of 16</span><div>${number(pair.query[f])} × ${number(pair.key[f])} = <strong>${number(pair.products[f])}</strong></div></div>
        <div class="ff-score-calculation"><div><span>Sum all 16 products</span><strong class="ff-dot-product">${number(pair.dotProduct)}</strong></div><span aria-hidden="true">÷ 4 →</span><div><span>Recorded scaled score [${r}, ${j}]</span><strong class="ff-scaled-score">${number(pair.score)}</strong></div></div>
        <p class="ff-caption">This one feature is only one term. The dot product uses all 16. Scaling by √16 = 4 is applied to every score in the matrix; displayed sums are reconstructed from the recorded features.</p>
      </div>
      <div class="ff-softmax-visual"><h3>Softmax acts on the whole receiving row</h3><p>All 95 scaled scores in row ${r} enter softmax together. The strip shows the 95 recorded weights, in source-token order.</p>
        <div class="ff-weight-strip" role="img" aria-label="All 95 attention weights for query ${r}, with source ${j} highlighted">${pair.weights.map((weight, i) => `<span class="ff-weight-cell${i === j ? " is-active" : ""}" data-source-index="${i}" data-value="${exact(weight)}" title="Source ${i} (${escape(reaction(i).reaction_id)}): ${exact(weight)}" style="--strength:${8 + weight / (Math.max(...pair.weights) || 1) * 92}%"></span>`).join("")}</div>
        <div class="ff-strip-labels"><span>Source 0</span><span>Source 94</span></div>
        <p class="ff-weight-result">Selected source ${j}: <strong>${percent(pair.weights[j])}</strong><span>All 95 weights sum to ${number(pair.weights.reduce((sum, weight) => sum + weight, 0))}.</span></p>
        <p class="ff-caption">Brighter cells indicate larger weights within this row. The highlighted cell is one weight, not a softmax calculated from one score.</p>
      </div>
    </figure>`;
  }

  function attentionView() {
    const { reaction: r, head: h } = state;
    const o = layer.operations;
    const sources = topSources(layer, r, h);
    if (state.source === null) state.source = sources[0].index;
    const j = state.source;
    const logits = o.scaled_logits[h][r];
    const evidence = softmaxEvidence(logits, j);
    const remaining = 1 - sources.reduce((sum, source) => sum + source.weight, 0);
    const terms = o.query[h][r].map((value, i) => [i, number(value), number(o.key[h][j][i]), number(value * o.key[h][j][i])]);
    return `${attentionDiagram(r, j, h)}
      <p class="ff-caption">Each head uses 16 features: 128 embedding features ÷ 8 heads. The model repeats this calculation for all 95 queries in each of the eight heads. Dividing scores by √16 = 4 keeps their scale from making softmax overly concentrated.</p>
      <div class="ff-attention-layout"><div>
      <p class="ff-visual-label">Five largest weights in this 95-source row <span>Head ${h + 1}</span></p>
      <div class="ff-attention-rows">${sources.map(source => `<button type="button" data-source="${source.index}" class="ff-attention-row${j === source.index ? " is-selected" : ""}" aria-pressed="${j === source.index}"><span>${escape(name(source.index))}</span><span class="ff-weight-track"><i style="width:${source.weight * 100}%"></i></span><strong>${percent(source.weight)}</strong></button>`).join("")}</div>
      <p class="ff-caption">Other 90 sources: ${percent(remaining)} in total. Weights use the full set of 95, without renormalizing this view.</p>
      </div><div class="ff-receiver"><span class="ff-x">Receiving reaction</span><strong>${escape(name(r))}</strong><code>${escape(reaction(r).reaction_id)}</code><p>Each row contributes information <em>to</em> this token.</p></div></div>
      ${equation(`<span class="ff-pair-score">Q<sub>${r}</sub> · K<sub>${j}</sub> / √16 = ${number(logits[j])}</span><span class="ff-softmax-row">All 95 scaled scores → softmax → weight for source ${j}: <strong>${percent(o.attention[h][r][j])}</strong></span>`, `Selected pair: ${escape(name(j))} → ${escape(name(r))}, one of 95 query–key comparisons for this receiver. These weights describe information flow inside the neural model, not biochemical connections.`)}
      <aside class="ff-insight"><strong>${state.layer === 0 ? "Layer 1 attention is condition-independent" : "Later attention can depend on the condition"}</strong><p>${state.layer === 0
        ? "Q and K come only from the learned initial embeddings. Changing the condition leaves these weights unchanged; the scalar values propagated through them still change."
        : "The preceding feed-forward block mixed nutrient scalars into embeddings. This layer compares those updated embeddings."}</p></aside>
      ${detail("attention", "How is this attention weight calculated?", `
        <p>LayerNorm normalizes each reaction’s 128-feature embedding before learned projections form Q, K, and V. Each projection is split across eight heads, leaving 16 features per head. V carries the information mixed in Step 3.</p>
        <p>For this selected pair only, sum all 16 query–key products and divide by √16 = 4. The same query is also compared with the other 94 keys. Softmax then normalizes the complete row of 95 scaled scores, including this one.</p>
        ${rowTable(["Receiving embedding (first four features)", "Before LayerNorm", "After LayerNorm"], layer.input_x[r].slice(0, 4).map((value, i) => [i, number(value), number(o.attention_norm[r][i])]))}
        ${rowTable(["Feature", "Query Q", "Source key K", "Q × K"], terms.slice(0, 4))}
        <details class="ff-subdetail"><summary>Remaining 12 dot-product terms</summary>${rowTable(["Feature", "Q", "K", "Q × K"], terms.slice(4))}</details>
        ${equation(`Recorded scaled logit = ${number(logits[j])}`, "The dot product uses all 16 features, not just the four previewed above.")}
        ${rowTable(["Stable softmax over 95 sources", "Value"], [["Subtract largest logit", number(evidence.maximum)], ["exp(selected logit − maximum)", number(evidence.numerator)], ["Sum exp(logit − maximum), all 95", number(evidence.denominator)], ["Recorded attention weight", number(o.attention[h][r][j])]])}
        <p>Each weight is exp(score − maximum) divided by the sum of those exponentials over all 95 sources. Subtracting the same maximum from every score improves numerical stability without changing the mathematical softmax result.</p>
        <p>The recorded float32 weight is displayed. The explanatory exponential terms use the same formula; values are rounded for readability.</p>`)}`;
  }

  function mixView() {
    const { reaction: r, head: h, feature: f } = state;
    const o = layer.operations;
    const evidence = scalarEvidence(layer, r, h, meta.head_weights[state.layer]);
    let body;
    if (state.channel === "scalar") {
      const heads = meta.head_weights[state.layer].map((alpha, head) => [head + 1, number(o.scalar_heads[head][r][0]), percent(alpha), number(alpha * o.scalar_heads[head][r][0])]);
      const sources = topSources(layer, r, h, 4);
      body = `<div class="ff-mix-flow"><div><span>One attention head</span><strong class="ff-c">${number(evidence.head)}</strong><small>Head ${h + 1}: weighted sum of 95 source scalars</small></div><span aria-hidden="true">→</span><div><span>Learned head combination</span><strong class="ff-c">${number(evidence.combined)}</strong><small>Sum of all eight head results × their learned weights</small></div></div>
        <div class="ff-residual"><span class="ff-residual-label">Residual addition</span>${number(evidence.before)} <span>+</span> ${number(evidence.combined)} <span>=</span> <strong class="ff-c">${number(evidence.after)}</strong><small>Input c + attention update = c after attention</small></div>
        <p class="ff-caption">This scalar is an intermediate state, not a nutrient bound or final flux prediction.</p>
        ${detail("scalar", "Inspect source contributions and learned head weights", `${rowTable(["Source (four strongest attention weights)", "Attention A", "Input c", "A × c"], sources.map(source => [escape(name(source.index)), number(source.weight), number(layer.input_c[source.index][0]), number(source.weight * layer.input_c[source.index][0])]))}
          <p>All 95 source terms contribute to head ${h + 1}’s recorded result ${number(evidence.head)}. A large attention weight can multiply a zero scalar.</p>
          ${rowTable(["Head", "Scalar result", "Learned α", "α × result"], heads)}
          <p>α = softmax(head_scores). These learned weights combine the scalar heads only. The current head has weight ${percent(evidence.alpha)}.</p>`)} `;
    } else {
      body = `<div class="ff-block-flow"><div><b>A × V</b><span>16 features / head</span></div><span aria-hidden="true">→</span><div><b>Concatenate 8 heads</b><span>128 features</span></div><span aria-hidden="true">→</span><div><b>Output projection</b><span>128 learned features</span></div></div>
        ${featureField("feature", "Embedding feature", f, 127)}
        <div class="ff-residual"><span class="ff-residual-label">Residual addition · feature ${f}</span>${number(layer.input_x[r][f])} <span>+</span> ${number(o.output_projection[r][f])} <span>=</span> <strong class="ff-x">${number(o.embedding_residual[r][f])}</strong><small>Input X[${r}, ${f}] + projected attention update = updated X[${r}, ${f}]</small></div>
        <p class="ff-caption">Embedding heads are concatenated and projected. They do not use the scalar channel’s learned α weighting.</p>
        ${detail("vector", "Inspect the selected head’s vector update", `${rowTable(["Head feature", "Weighted value A × V"], o.weighted_values[h][r].slice(0, 4).map((value, i) => [i, number(value)]))}<p>Four of the head’s 16 features are previewed. All eight heads contribute to the 128-feature output projection.</p>`)} `;
    }
    return `<div class="ff-channel-switch" role="group" aria-label="Channel to inspect"><button type="button" data-channel="scalar" aria-pressed="${state.channel === "scalar"}"><span class="ff-c">c</span> Scalar channel</button><button type="button" data-channel="embedding" aria-pressed="${state.channel === "embedding"}"><span class="ff-x">X</span> Embedding channel</button></div>${body}`;
  }

  function feedforwardView() {
    const { reaction: r, hidden: k } = state;
    const o = layer.operations;
    const projection = meta.scalar_projection[state.layer];
    return `<div class="ff-join"><span class="ff-x">Row of X · 128 features</span><span>+</span><span class="ff-c">c · 1 scalar</span><span>→</span><strong>129 joined features</strong></div>
      <div class="ff-ffn"><p>Same network applied separately to every reaction</p><div class="ff-block-flow"><div><b>LayerNorm</b><span>129 features</span></div><span aria-hidden="true">→</span><div><b>Linear → GELU</b><span>640 hidden features</span></div><span aria-hidden="true">→</span><div><b>Linear</b><span>129 output features</span></div></div></div>
      <div class="ff-residual"><span class="ff-residual-label">Follow the scalar feature</span>${number(o.scalar_residual[r][0])} <span>+</span> ${number(o.linear2[r][128])} <span>=</span> <strong class="ff-c">${number(o.output_c[r][0])}</strong><small>Joined input c + feed-forward correction = layer ${state.layer + 1} scalar</small></div>
      <p class="ff-caption">The residual is added to all 129 features, then split back into a row of X (128) and c (1). ${state.layer < 2 ? "This layer’s scalar is still intermediate." : "At layer 3, the resulting scalar is the final flux prediction."}</p>
      ${detail("hidden", "Follow one hidden feature through the calculation", `${featureField("hidden", "Hidden feature", k, 639)}
        ${rowTable(["Recorded operation", "Value"], [["Scalar feature after joined LayerNorm", number(o.feedforward_norm[r][128])], ["First linear output z", number(o.linear1[r][k])], ["GELU(z)", number(o.gelu[r][k])], ["Weight into scalar output", number(projection.weights[k])], ["This feature’s contribution", number(o.gelu[r][k] * projection.weights[k])], ["Scalar output bias", number(projection.bias)], ["Recorded sum of all 640 contributions + bias", number(o.linear2[r][128])]])}
        ${equation("GELU(z) = ½z [1 + erf(z / √2)]", "Dropout is disabled for these eval-mode traces. The second linear map uses all 640 activated hidden features, then the residual restores the joined input.")}
        <p>LayerNorm normalizes each joined row of X and its scalar c together. The feed-forward network can therefore pass nutrient information from the scalar into the updated embedding.</p>`)}`;
  }

  function layersView() {
    const values = trajectory(overview, state.reaction);
    const low = Math.min(0, ...values), high = Math.max(0, ...values);
    const span = high - low || 1;
    const y = value => 170 - (value - low) / span * 115;
    const xs = [65, 240, 415, 590];
    const labels = ["Initialized", "Layer 1", "Layer 2", "Layer 3 · final"];
    return `<figure class="ff-trajectory"><svg viewBox="0 0 660 235" role="img" aria-label="Recorded scalar states for ${escape(reaction(state.reaction).reaction_id)}: ${values.map(value => format(value)).join(", ")}"><line class="ff-chart-baseline" x1="45" x2="620" y1="${y(0)}" y2="${y(0)}"/><polyline class="ff-chart-line" points="${values.map((value, i) => `${xs[i]},${y(value)}`).join(" ")}"/>${values.map((value, i) => `<circle class="${i === 3 ? "ff-final-dot" : "ff-state-dot"}" cx="${xs[i]}" cy="${y(value)}" r="7"/><text x="${xs[i]}" y="${y(value) - 17}" text-anchor="middle">${format(value, 4)}</text><text class="ff-axis-label" x="${xs[i]}" y="213" text-anchor="middle">${labels[i]}</text>`).join("")}</svg><figcaption>A recorded state after each layer. The line connects stages of the network; it is not a trajectory over biological time.</figcaption></figure>
      <div class="ff-state-list">${values.map((value, i) => `<div class="${i === 3 ? "is-final" : ""}"><span>${labels[i]}</span><strong>${number(value)}</strong><small>${i === 0 ? "Initial context" : i === 3 ? "Final prediction" : "Intermediate scalar"}</small></div>`).join("")}</div>
      <div class="ff-insight"><strong>Both streams move forward</strong><p>Each layer passes its updated 128-feature embedding and scalar to the next. Layer 1 attention is fixed across conditions; later attention can respond to nutrient context through the updated embeddings.</p></div>
      ${detail("states", "Inspect the recorded states for this reaction", rowTable(["Stage", "Scalar c", "Interpretation"], values.map((value, i) => [labels[i], number(value), i === 3 ? "Final flux prediction" : "Intermediate / initial state"])) + `<p>${unitNote()} Early-layer states have not been validated as standalone flux predictions.</p>`)}`;
  }

  function predictionsView() {
    const prediction = overview.states[2][state.reaction];
    const target = overview.targets[state.reaction];
    const maximum = Math.max(Math.abs(prediction), Math.abs(target), 0.000001);
    const chart = (label, value, cls) => `<div class="ff-comparison-row"><span>${label}</span><div class="ff-diverging"><i class="${cls}" style="left:${value < 0 ? 50 - Math.abs(value) / maximum * 48 : 50}%;width:${Math.abs(value) / maximum * 48}%"></i></div><strong>${number(value)}</strong></div>`;
    const filtered = meta.reactions.filter(r => `${r.reaction_id} ${r.name}`.toLowerCase().includes(state.search.toLowerCase()));
    const pages = Math.max(1, Math.ceil(filtered.length / 6));
    state.page = Math.min(state.page, pages - 1);
    const rows = filtered.slice(state.page * 6, state.page * 6 + 6).map(r => [
      `<button class="ff-reaction-link" data-reaction="${r.index}" type="button">${escape(r.reaction_id)}</button>`,
      number(overview.states[2][r.index]), number(overview.targets[r.index]), number(Math.abs(overview.states[2][r.index] - overview.targets[r.index])),
    ]);
    return `<div class="ff-results-heading"><span>Selected reaction</span><h3>${escape(name(state.reaction))}</h3><code>${escape(reaction(state.reaction).reaction_id)}</code></div>
      <div class="ff-comparison" aria-label="Signed prediction and simulation target">${chart("FluxFormer", prediction, "ff-predicted")}${chart("Simulation target", target, "ff-target")}<div class="ff-comparison-axis"><span>Negative direction</span><span>0</span><span>Positive direction</span></div></div>
      <div class="ff-result-values">${card("Final predicted flux", prediction, "ff-predicted-value")}${card("Simulated target flux", target)}${card("Absolute error", Math.abs(prediction - target))}</div>
      <p class="ff-caption">${unitNote()} Predictions and targets come from the same condition. Targets are simulated pFBA results, not measurements.</p>
      ${detail("results", "Explore all 95 reaction predictions", `<label class="ff-field" for="ff-search"><span>Find a reaction by ID or name</span><input type="search" id="ff-search" value="${escape(state.search)}" placeholder="e.g. PFK or glucose"></label>
        ${rows.length ? rowTable(["Reaction", "Predicted", "Target", "Absolute error"], rows) : "<p>No matching reactions.</p>"}
        <div class="ff-pagination"><button type="button" class="ff-button" data-page="-1"${state.page === 0 ? " disabled" : ""}>← Previous six</button><span>${filtered.length} reactions · ${state.page + 1} / ${pages}</span><button type="button" class="ff-button" data-page="1"${state.page + 1 >= pages ? " disabled" : ""}>Next six →</button></div>`)}
      <div class="ff-insight"><strong>A surrogate of the simulator</strong><p>These four examples illustrate the calculation, not generalization performance. A final prediction can differ from its target and can violate metabolic constraints, which the network does not enforce.</p></div>`;
  }

  async function render({ keepDetails = false, focusTitle = false } = {}) {
    const version = ++revision;
    const open = keepDetails ? [...$("stage").querySelectorAll("details[open][data-key]")].map(d => d.dataset.key) : [];
    const active = document.activeElement;
    const restoreFocus = keepDetails && $("stage").contains(active) ? active.id : "";
    const buttonAttribute = keepDetails && $("stage").contains(active) ?
      ["data-source", "data-reaction", "data-channel", "data-page"].find(attribute => active.hasAttribute(attribute)) : null;
    const restoreButton = buttonAttribute ? `[${buttonAttribute}="${active.getAttribute(buttonAttribute)}"]` : "";
    const caret = restoreFocus && active.type === "search" ? active.selectionStart : null;
    const sample = meta.samples.find(item => item.row === state.sample);
    $("step-label").textContent = `Step ${state.step + 1} of 6`;
    $("stage-title").textContent = titles[state.step];
    $("stage-description").textContent = descriptions[state.step];
    $("reaction-label").textContent = state.step === 1 ? "Receiving reaction (query)" : "Inspect reaction";
    $("layer-field").hidden = ![1, 2, 3].includes(state.step);
    $("head-field").hidden = ![1, 2].includes(state.step);
    $("previous").disabled = state.step === 0;
    $("next").disabled = state.step === 5;
    $("next").textContent = state.step === 5 ? "Walkthrough complete" : "Next →";
    $("progress").textContent = `${state.step + 1} / 6`;
    root.querySelectorAll("[data-step]").forEach(link => {
      if (Number(link.dataset.step) === state.step) link.setAttribute("aria-current", "step");
      else link.removeAttribute("aria-current");
    });
    const needsLayer = [1, 2, 3].includes(state.step);
    const alreadyLoaded = (!needsLayer || store.layers.has(`${sample.row}:${state.layer}`)) && store.overviews.has(sample.row);
    if (!alreadyLoaded) {
      $("stage").innerHTML = `<p class="ff-loading" role="status">Loading ${needsLayer ? `layer ${state.layer + 1}’s full calculation` : "condition"}…</p>`;
      $("condition-note").textContent = `Condition row ${sample.row}`;
    }
    $("stage").setAttribute("aria-busy", "true");
    try {
      const loadedOverview = await store.overview(sample);
      const loadedLayer = needsLayer ? await store.layer(sample, state.layer) : null;
      if (version !== revision) return;
      overview = loadedOverview;
      layer = loadedLayer;
      $("condition-note").textContent = `Row ${sample.row} · oxygen bound ${format(overview.constraints[18])} · inputs held fixed`;
      $("download").href = new URL(sample.overview, store.url).href;
      $("stage").innerHTML = [initializeView, attentionView, mixView, feedforwardView, layersView, predictionsView][state.step]();
      for (const key of open) {
        const element = $("stage").querySelector(`details[data-key="${key}"]`);
        if (element) element.open = true;
      }
      if (restoreFocus && document.getElementById(restoreFocus)) {
        document.getElementById(restoreFocus).focus({ preventScroll: true });
        if (caret !== null) document.getElementById(restoreFocus).setSelectionRange(caret, caret);
      } else if (restoreButton) {
        $("stage").querySelector(restoreButton)?.focus({ preventScroll: true });
      }
      if (focusTitle) $("stage-title").focus({ preventScroll: true });
    } catch (error) {
      if (version !== revision) return;
      $("stage").innerHTML = `<div class="ff-notice" role="alert"><p>${escape(error.message)}</p><button type="button" class="ff-button" data-reload>Retry calculation</button><p>Use the steps above to return to initialization or final predictions.</p></div>`;
    } finally {
      if (version === revision) $("stage").setAttribute("aria-busy", "false");
    }
  }

  $("retry").addEventListener("click", initialize);
  $("sample").addEventListener("change", () => { state.sample = Number($("sample").value); state.source = null; render(); });
  $("reaction").addEventListener("change", () => { state.reaction = Number($("reaction").value); state.source = null; render(); });
  $("layer").addEventListener("change", () => { state.layer = Number($("layer").value); state.source = null; render(); });
  $("head").addEventListener("change", () => { state.head = Number($("head").value); state.source = null; render(); });
  $("previous").addEventListener("click", () => { if (state.step > 0) location.hash = steps[state.step - 1]; });
  $("next").addEventListener("click", () => { if (state.step < 5) location.hash = steps[state.step + 1]; });
  window.addEventListener("hashchange", () => {
    state.step = Math.max(0, steps.indexOf(location.hash.slice(1)));
    if (meta) render({ focusTitle: true });
  });
  $("stage").addEventListener("click", event => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.hasAttribute("data-reaction")) {
      state.reaction = Number(button.dataset.reaction);
      $("reaction").value = state.reaction;
      state.source = null;
    } else if (button.hasAttribute("data-source")) state.source = Number(button.dataset.source);
    else if (button.hasAttribute("data-channel")) state.channel = button.dataset.channel;
    else if (button.hasAttribute("data-page")) state.page += Number(button.dataset.page);
    else if (!button.hasAttribute("data-reload")) return;
    render({ keepDetails: true });
  });
  $("stage").addEventListener("change", event => {
    const { id, value } = event.target;
    if (id === "ff-source") state.source = Number(value);
    else if (id === "ff-nutrient") state.nutrient = Number(value);
    else if (id === "ff-feature" || id === "ff-hidden" || id === "ff-pair-feature") {
      const field = id === "ff-feature" ? "feature" : id === "ff-hidden" ? "hidden" : "pairFeature";
      const maximum = id === "ff-feature" ? 127 : id === "ff-hidden" ? 639 : 15;
      if (!Number.isInteger(Number(value)) || Number(value) < 0 || Number(value) > maximum || value === "") {
        event.target.value = state[field];
        return;
      }
      state[field] = Number(value);
    } else return;
    render({ keepDetails: true });
  });
  $("stage").addEventListener("input", event => {
    if (event.target.id === "ff-search") {
      state.search = event.target.value;
      state.page = 0;
      render({ keepDetails: true });
    }
  });
  initialize();
}
