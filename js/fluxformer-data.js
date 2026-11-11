// Read verified, precomputed traces. No model inference or input editing.
export const steps = ["initialize", "attention", "mix", "feedforward", "layers", "predictions"];

export function format(value, digits = 5) {
  if (!Number.isFinite(value)) throw new Error("Non-finite trace value");
  if (value === 0) return "0";
  if (Math.abs(value) < 0.0001) return value.toExponential(3).replace("e-", "e−");
  return new Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(value);
}

export function topSources(layer, reaction, head, limit = 5) {
  return layer.operations.attention[head][reaction]
    .map((weight, index) => ({ index, weight }))
    .sort((a, b) => b.weight - a.weight || a.index - b.index).slice(0, limit);
}

export function softmaxEvidence(logits, index) {
  const maximum = Math.max(...logits);
  const denominator = logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
  return { maximum, denominator, numerator: Math.exp(logits[index] - maximum) };
}

export function attentionPairEvidence(layer, reaction, source, head) {
  const query = layer.operations.query[head][reaction];
  const key = layer.operations.key[head][source];
  const products = query.map((value, i) => value * key[i]);
  return {
    query, key, products, dotProduct: products.reduce((sum, value) => sum + value, 0),
    scale: Math.sqrt(query.length),
    score: layer.operations.scaled_logits[head][reaction][source],
    weights: layer.operations.attention[head][reaction],
  };
}

export function valueMixEvidence(layer, reaction, source, head, feature) {
  const weights = layer.operations.attention[head][reaction];
  const values = layer.operations.value[head].map(vector => vector[feature]);
  const products = weights.map((weight, index) => weight * values[index]);
  return {
    weights, values, products, sum: products.reduce((total, value) => total + value, 0),
    weight: weights[source], value: values[source], contribution: products[source],
    output: layer.operations.weighted_values[head][reaction][feature],
    headOutput: layer.operations.weighted_values[head][reaction],
  };
}

export function scalarEvidence(layer, reaction, head, weights) {
  const operations = layer.operations;
  return {
    before: layer.input_c[reaction][0],
    head: operations.scalar_heads[head][reaction][0],
    alpha: weights[head],
    combined: operations.scalar_combined[reaction][0],
    after: operations.scalar_residual[reaction][0],
  };
}

export function trajectory(overview, reaction) {
  return [overview.initial_c[reaction], ...overview.states.map(state => state[reaction])];
}

export class TraceStore {
  constructor(manifestURL, fetcher = (...args) => globalThis.fetch(...args)) {
    this.url = new URL(manifestURL);
    this.fetcher = fetcher;
    this.overviews = new Map();
    this.layers = new Map();
  }

  async read(filename) {
    const response = await this.fetcher(new URL(filename, this.url));
    if (!response.ok) throw new Error(`Trace data could not be loaded (HTTP ${response.status}).`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
      if (typeof DecompressionStream === "undefined") {
        throw new Error("Layer details need a browser with gzip decompression support. Initialization, layer states, and predictions are still available.");
      }
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      return JSON.parse(await new Response(stream).text());
    }
    // Handles ordinary JSON and hosts that already applied Content-Encoding.
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  async metadata() {
    if (!this.meta) {
      this.meta = await this.read(this.url.href);
      if (this.meta.schema_version !== "1.0.0" || this.meta.reactions.length !== 95 ||
          this.meta.inputs.length !== 20 || !this.meta.verification.passed) {
        this.meta = null;
        throw new Error("The trace manifest does not match the verified core model.");
      }
    }
    return this.meta;
  }

  overview(sample) {
    if (!this.overviews.has(sample.row)) {
      const request = this.read(sample.overview).then(data => {
        if (data.row !== sample.row || data.targets.length !== 95 || data.states.length !== 3 ||
            data.states.some(state => state.length !== 95) || !data.verification.passed) {
          throw new Error("Unexpected condition data.");
        }
        return data;
      }).catch(error => { this.overviews.delete(sample.row); throw error; });
      this.overviews.set(sample.row, request);
    }
    return this.overviews.get(sample.row);
  }

  layer(sample, index) {
    const key = `${sample.row}:${index}`;
    if (!this.layers.has(key)) {
      // Keep at most two decoded layers in memory, especially on mobile.
      if (this.layers.size >= 2) this.layers.delete(this.layers.keys().next().value);
      const request = this.read(sample.layers[index]).then(data => {
        if (data.sample_row !== sample.row || data.index !== index || data.input_c.length !== 95 ||
            data.operations.attention.length !== 8 ||
            data.operations.attention.some(head => head.length !== 95 || head.some(row => row.length !== 95))) {
          throw new Error("Unexpected full-vocabulary layer data.");
        }
        return data;
      }).catch(error => { this.layers.delete(key); throw error; });
      this.layers.set(key, request);
    }
    return this.layers.get(key);
  }
}
