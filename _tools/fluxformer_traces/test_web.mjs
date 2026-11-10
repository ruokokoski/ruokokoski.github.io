import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { TraceStore, format, topSources, scalarEvidence, softmaxEvidence, trajectory } from "../../js/fluxformer-data.js";

const base = new URL("../../data/fluxformer/manifest.json", import.meta.url);
const localFetch = async url => {
  try { return new Response(await readFile(fileURLToPath(url))); }
  catch { return new Response("not found", { status: 404 }); }
};
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) <= 2e-5 + 2e-5 * Math.abs(expected), `${actual} != ${expected}`);

test("browser data helpers preserve all conditions, sources, heads, and scalar states", async () => {
  const store = new TraceStore(base, localFetch);
  const meta = await store.metadata();
  assert.equal(meta.reactions.length, 95);
  assert.equal(meta.samples.length, 4);
  for (const sample of meta.samples) {
    const overview = await store.overview(sample);
    for (const r of [12, 27, 35, 94]) {
      assert.deepEqual(trajectory(overview, r), [overview.initial_c[r], ...overview.states.map(states => states[r])]);
    }
    for (let i = 0; i < 3; i++) {
      const layer = await store.layer(sample, i);
      for (const r of [12, 27, 35, 94]) {
        close(layer.operations.output_c[r][0], overview.states[i][r]);
        for (const h of [0, 7]) {
          const sources = topSources(layer, r, h);
          assert.equal(sources.length, 5);
          assert.ok(sources.every((item, j) => j === 0 || sources[j - 1].weight >= item.weight));
          assert.ok(sources.reduce((sum, item) => sum + item.weight, 0) < 1);
          const selected = sources[0].index;
          const softmax = softmaxEvidence(layer.operations.scaled_logits[h][r], selected);
          close(softmax.numerator / softmax.denominator, layer.operations.attention[h][r][selected]);
          const scalar = scalarEvidence(layer, r, h, meta.head_weights[i]);
          close(scalar.before + scalar.combined, scalar.after);
          close(layer.operations.attention[h][r].reduce((sum, value, j) => sum + value * layer.input_c[j][0], 0), scalar.head);
        }
      }
      assert.ok(store.layers.size <= 2, "decoded layer cache stays bounded");
    }
  }
});

test("failed layer downloads can be retried and do not leave failed cache entries", async () => {
  let fail = true;
  const store = new TraceStore(base, async url => {
    if (String(url).includes("-layer-") && fail) return new Response("unavailable", { status: 503 });
    return localFetch(url);
  });
  const meta = await store.metadata();
  await assert.rejects(store.layer(meta.samples[0], 0), /503/);
  assert.equal(store.layers.size, 0);
  fail = false;
  assert.equal((await store.layer(meta.samples[0], 0)).input_c.length, 95);
});

test("missing decompression support preserves access to lightweight overviews", async () => {
  const store = new TraceStore(base, localFetch);
  const meta = await store.metadata();
  const original = globalThis.DecompressionStream;
  try {
    globalThis.DecompressionStream = undefined;
    await assert.rejects(store.layer(meta.samples[0], 0), /decompression support/);
    assert.equal((await store.overview(meta.samples[0])).targets.length, 95);
  } finally { globalThis.DecompressionStream = original; }
});

test("formatting keeps small signed values visible and rejects non-finite data", () => {
  assert.equal(format(0), "0");
  assert.match(format(-0.0000001), /−7/);
  assert.equal(format(-0.62), "-0.62");
  assert.throws(() => format(NaN), /Non-finite/);
});

test("default fetch keeps the browser Window receiver", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = function (url) {
      assert.equal(this, globalThis);
      return localFetch(url);
    };
    const store = new TraceStore(base);
    assert.equal((await store.metadata()).reactions.length, 95);
  } finally { globalThis.fetch = original; }
});
