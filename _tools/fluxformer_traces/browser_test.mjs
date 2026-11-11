// Optional development dependency: Playwright 1.51.1 with its installed browsers.
// The website itself has no npm dependency.
import assert from "node:assert/strict";
import { readFile, mkdir } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";

const driver = await import(process.env.FLUXFORMER_PLAYWRIGHT_MODULE || "playwright");
const base = process.env.FLUXFORMER_PREVIEW_URL || "http://127.0.0.1:4175";
const screenshots = process.env.FLUXFORMER_SCREENSHOTS || "/tmp/fluxformer-browser-screenshots";
await mkdir(screenshots, { recursive: true });
const dataRoot = new URL("../../data/fluxformer/", import.meta.url);
const json = async filename => {
  const bytes = await readFile(new URL(filename, dataRoot));
  return JSON.parse(filename.endsWith(".gz") ? gunzipSync(bytes) : bytes);
};
const meta = await json("manifest.json");
const sample = row => meta.samples.find(item => item.row === row);
const errors = [];
let assertions = 0;
function check(value, message) { assert.ok(value, message); assertions++; }
const busy = page => page.waitForFunction(() => document.getElementById("ff-stage").getAttribute("aria-busy") === "false");
async function step(page, index) {
  await page.locator(`[data-step="${index}"]`).click();
  await page.waitForFunction(i => document.getElementById("ff-step-label").textContent === `Step ${i + 1} of 6`, index);
  await busy(page);
  check(await page.locator(`[data-step="${index}"]`).getAttribute("aria-current") === "step", "current architecture component is identified");
}
async function change(page, id, value) {
  await page.locator(`#ff-${id}`).selectOption(String(value));
  await busy(page);
}
async function values(locator) {
  return locator.evaluateAll(elements => elements.map(element => Number(element.title)));
}
async function fits(page, label) {
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `no page overflow: ${label}`);
}

async function checkMatrixDiagram(page, expected, receiver, source, head) {
  const diagram = page.locator(".ff-matrix-walkthrough");
  check(await diagram.getAttribute("data-query") === String(receiver) &&
        await diagram.getAttribute("data-key") === String(source) &&
        await diagram.getAttribute("data-head") === String(head) &&
        await diagram.getAttribute("data-layer") === String(expected.index), "matrix highlights follow receiver, source, head, and layer");
  const vectors = await diagram.locator(".ff-pair-vector .ff-pair-cell").evaluateAll(cells => cells.map(cell => Number(cell.dataset.value)));
  check(JSON.stringify(vectors) === JSON.stringify([...expected.operations.query[head][receiver], ...expected.operations.key[head][source]]), "both 16-feature vectors equal the recorded Q and K");
  const weights = await diagram.locator(".ff-weight-cell").evaluateAll(cells => cells.map(cell => Number(cell.dataset.value)));
  check(JSON.stringify(weights) === JSON.stringify(expected.operations.attention[head][receiver]), "softmax strip preserves all 95 recorded weights without subset normalization");
  check(await diagram.locator(".ff-weight-cell.is-active").getAttribute("data-source-index") === String(source), "softmax strip highlights the chosen source");
  check((await values(diagram.locator(".ff-scaled-score .ff-number")))[0] === expected.operations.scaled_logits[head][receiver][source], "diagram uses the original recorded scaled score");
}

async function checkValueDiagram(page, expected, receiver, head, feature) {
  const diagram = page.locator(".ff-value-matrix");
  const source = Number(await diagram.getAttribute("data-source"));
  check(await diagram.getAttribute("data-receiver") === String(receiver) &&
        await diagram.getAttribute("data-head") === String(head) &&
        await diagram.getAttribute("data-layer") === String(expected.index) &&
        await diagram.getAttribute("data-feature") === String(feature), "A × V highlights follow receiver, head, layer, and output feature");
  const weights = expected.operations.attention[head][receiver];
  const column = expected.operations.value[head].map(vector => vector[feature]);
  for (const [kind, reference] of [["attention", weights], ["values", column], ["products", weights.map((value, i) => value * column[i])]]) {
    const actual = await diagram.locator(`.ff-value-strip-${kind} .ff-value-cell`).evaluateAll(cells => cells.map(cell => Number(cell.dataset.value)));
    check(JSON.stringify(actual) === JSON.stringify(reference), `all 95 A × V ${kind} entries are exact and in source order`);
  }
  check(await diagram.locator(`.ff-value-cell.is-active[data-source-index="${source}"]`).count() === 3, "source selection links weight, value, and contribution strips");
  check((await values(diagram.locator(".ff-value-output .ff-number")))[0] === expected.operations.weighted_values[head][receiver][feature], "head-output result is the original recorded value");
  const output = await diagram.locator(".ff-pair-cell").evaluateAll(cells => cells.map(cell => Number(cell.dataset.value)));
  check(JSON.stringify(output) === JSON.stringify(expected.operations.weighted_values[head][receiver]), "head-output strip preserves all 16 recorded features");
}

async function checkInitialMatrices(page, overview, receiver, feature) {
  const diagram = page.locator(".ff-initial-matrices");
  check(await diagram.getAttribute("data-reaction") === String(receiver) &&
        await diagram.getAttribute("data-feature") === String(feature), "X and c share the selected reaction row and X feature");
  const row = await diagram.locator(".ff-initial-feature").evaluateAll(cells => cells.map(cell => Number(cell.dataset.value)));
  check(JSON.stringify(row) === JSON.stringify(meta.initial_x[receiver]), "highlighted X row preserves all 128 learned features");
  const scalars = await diagram.locator(".ff-initial-c-cell").evaluateAll(cells => cells.map(cell => Number(cell.dataset.value)));
  check(JSON.stringify(scalars) === JSON.stringify(overview.initial_c), "c column preserves all 95 initialized scalars");
  const slots = await diagram.locator(".ff-initial-c-cell").evaluateAll(cells => cells.map(cell => cell.dataset.input === "true"));
  check(slots.filter(Boolean).length === 20 && slots.every((input, i) => input === (meta.reactions[i].input_index !== null)), "all 20 input slots and 75 non-input slots retain the exact mapping");
  const selected = await values(diagram.locator(".ff-initial-values .ff-number"));
  check(selected[0] === meta.initial_x[receiver][feature] && selected[1] === overview.initial_c[receiver], "selected X cell and matching c scalar are exact");
  check(await page.locator(`.ff-vector-full .ff-feature.is-active[data-feature="${feature}"]`).count() === 1, "full-vector inspector highlights the same feature as X");
}

async function clickInitialMatrix(page, matrix, receiver, feature = 0) {
  const svg = page.locator(".ff-initial-matrix-svg");
  await svg.scrollIntoViewIfNeeded();
  const point = await svg.evaluate((element, input) => {
    const box = element.querySelector(input.matrix === "X" ? ".ff-initial-x-body" : ".ff-initial-c-body").getBBox();
    const local = new DOMPoint(box.x + (input.matrix === "X" ? (input.feature + .5) / 128 : .5) * box.width,
      box.y + (input.receiver + .5) / 95 * box.height);
    const screen = local.matrixTransform(element.getScreenCTM());
    return { x: screen.x, y: screen.y };
  }, { matrix, receiver, feature });
  await page.mouse.click(point.x, point.y);
  await busy(page);
}

for (const engine of ["chromium", "webkit"]) {
  const browser = await driver[engine].launch({ headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(`${engine}: ${error.message}`));
    const requests = [];
    page.on("request", request => requests.push(request.url()));
    await page.goto(`${base}/fluxformer/`);
    await page.waitForSelector(".ff-initial-matrices");
    check((await page.locator(".site-nav a").allTextContents()).join("|") === "Home|FluxFormer|Research|Blog", "navigation order");
    check(await page.locator(".site-nav a[aria-current=page]").textContent() === "FluxFormer", "active navigation");
    check(!requests.some(url => url.endsWith(".json.gz")), "initialization does not download full layers");
    check(await page.locator(".ff-token").count() === 0, "X/c matrix overview replaces the six example cards");
    check(await page.locator("#ff-layer-field").isHidden(), "layer control is contextual");
    await page.locator('[data-key="nutrients"] > summary').click();
    await change(page, "nutrient", 18);
    const overview722 = await json(sample(722).overview);
    check((await values(page.locator('[data-key="nutrients"] .ff-number')))[0] === overview722.constraints[18], "displayed oxygen bound is exact");
    await checkInitialMatrices(page, overview722, 12, 0);
    await change(page, "reaction", 0);
    await checkInitialMatrices(page, overview722, 0, 0);
    check(await page.locator(".ff-inspection > header code").textContent() === "ACALD", "selected reaction appears in its own panel");
    await change(page, "reaction", 25);
    await page.locator("#ff-feature").fill("127");
    await page.locator("#ff-feature").press("Tab");
    await busy(page);
    await checkInitialMatrices(page, overview722, 25, 127);
    await page.locator("#ff-feature").fill("0");
    await page.locator("#ff-feature").press("Tab");
    await busy(page);
    check((await values(page.locator(".ff-initial-scalar .ff-number")))[0] === overview722.initial_c[25], "separate panel displays the selected input scalar exactly");
    const features = await page.locator(".ff-vector-full .ff-feature").evaluateAll(bars => bars.map(bar => ({
      index: Number(bar.dataset.feature), value: Number(bar.dataset.value),
      sign: bar.classList.contains("negative") ? -1 : bar.classList.contains("positive") ? 1 : 0,
      color: getComputedStyle(bar.querySelector("i")).backgroundColor,
    })));
    check(features.length === 128 && features.every((feature, i) => feature.index === i && feature.value === meta.initial_x[25][i]), "selected panel displays all 128 exact features in order");
    check(features.every(feature => feature.sign === Math.sign(feature.value)), "feature sign matches the recorded number");
    const blue = features.find(feature => feature.sign < 0).color.match(/\d+/g).map(Number);
    const red = features.find(feature => feature.sign > 0).color.match(/\d+/g).map(Number);
    check(blue[2] > blue[0] && red[0] > red[2], "negative bars are blue and positive bars are red");
    await page.locator('[data-key="embedding"] > summary').click();
    check((await values(page.locator('[data-key="embedding"] .ff-equation .ff-number')))[0] === meta.initial_x[25][0], "embedding inspector follows the separate selection");
    await clickInitialMatrix(page, "X", 27, 12);
    await checkInitialMatrices(page, overview722, 27, 12);
    check(await page.locator("#ff-reaction").inputValue() === "27" && await page.locator(".ff-inspection > header code").textContent() === "EX_glc__D_e", "clicking X updates the reaction selector and inspection panel");
    check(await page.locator("#ff-feature").inputValue() === "12", "clicking X selects its feature column");
    await clickInitialMatrix(page, "c", 12);
    await checkInitialMatrices(page, overview722, 12, 12);
    check(await page.locator("#ff-feature").inputValue() === "12", "clicking c changes the reaction while retaining the X feature");
    await change(page, "sample", 870);
    await checkInitialMatrices(page, await json(sample(870).overview), 12, 12);
    await change(page, "sample", 722);
    await page.screenshot({ path: `${screenshots}/${engine}-initial.png`, fullPage: true });
    await page.locator(".ff-initial-matrices").screenshot({ path: `${screenshots}/${engine}-initial-matrices.png` });
    await page.locator(".ff-inspection").screenshot({ path: `${screenshots}/${engine}-selected-vector.png` });

    await step(page, 1);
    check(await page.locator(".ff-attention-row").count() === 5, "attention shows five sources");
    const firstLayer = await json(sample(722).layers[0]);
    const firstSource = Number(await page.locator(".ff-attention-row.is-selected").getAttribute("data-source"));
    await checkMatrixDiagram(page, firstLayer, 12, firstSource, 0);
    await page.locator("#ff-pair-feature").fill("15");
    await page.locator("#ff-pair-feature").press("Tab");
    await busy(page);
    const productValues = await values(page.locator(".ff-feature-product .ff-number"));
    check(productValues[0] === firstLayer.operations.query[0][12][15] && productValues[1] === firstLayer.operations.key[0][firstSource][15] && productValues[2] === productValues[0] * productValues[1], "selected feature shows its actual query, key, and product");
    check(await page.locator('.ff-pair-cell.is-active[data-feature="15"]').count() === 2, "feature selection links both vector strips");
    await page.locator("#ff-pair-feature").fill("16");
    await page.locator("#ff-pair-feature").press("Tab");
    check(await page.locator("#ff-pair-feature").inputValue() === "15", "feature inspection stays within the head's 16 features");
    check((await values(page.locator(".ff-equation .ff-number")))[0] === firstLayer.operations.scaled_logits[0][12][firstSource], "selected logit is exact");
    const firstWeights = await page.locator(".ff-attention-row strong").allTextContents();
    await page.locator(".ff-attention-row").nth(1).focus();
    await page.keyboard.press("Enter");
    await busy(page);
    check(await page.evaluate(() => document.activeElement.matches(".ff-attention-row.is-selected")), "source selection preserves keyboard focus");
    await page.locator('[data-key="attention"] > summary').click();
    await change(page, "source", 94);
    await checkMatrixDiagram(page, firstLayer, 12, 94, 0);
    check((await values(page.locator(".ff-equation .ff-number")))[0] === firstLayer.operations.scaled_logits[0][12][94], "any of 95 sources can be inspected");
    check(await page.locator('[data-key="attention"]').getAttribute("open") !== null, "details remain open after local selection");
    await change(page, "reaction", 27);
    await checkMatrixDiagram(page, firstLayer, 27, Number(await page.locator(".ff-matrix-walkthrough").getAttribute("data-key")), 0);
    await change(page, "reaction", 12);
    await change(page, "sample", 870);
    check(JSON.stringify(await page.locator(".ff-attention-row strong").allTextContents()) === JSON.stringify(firstWeights), "first-layer attention display stays fixed across conditions");
    await change(page, "head", 7);
    await change(page, "layer", 1);
    const attentionLayer = await json(sample(870).layers[1]);
    await checkMatrixDiagram(page, attentionLayer, 12, Number(await page.locator(".ff-matrix-walkthrough").getAttribute("data-key")), 7);
    check((await page.locator(".ff-insight").innerText()).includes("Later attention"), "later-layer context explanation");
    await page.screenshot({ path: `${screenshots}/${engine}-attention.png`, fullPage: true });
    await page.locator(".ff-matrix-walkthrough").screenshot({ path: `${screenshots}/${engine}-matrix-diagram.png` });

    await step(page, 2);
    const layer = await json(sample(870).layers[1]);
    const scalarNumbers = await values(page.locator(".ff-residual .ff-number"));
    check(scalarNumbers[0] === layer.input_c[12][0] && scalarNumbers[2] === layer.operations.scalar_residual[12][0], "scalar residual uses exact recorded values");
    await page.locator('[data-channel="embedding"]').click();
    await busy(page);
    await checkValueDiagram(page, layer, 12, 7, 0);
    await page.locator("#ff-value-feature").fill("15");
    await page.locator("#ff-value-feature").press("Tab");
    await busy(page);
    await checkValueDiagram(page, layer, 12, 7, 15);
    const beforeSourceChange = await values(page.locator(".ff-value-output .ff-number"));
    await change(page, "source", 94);
    await checkValueDiagram(page, layer, 12, 7, 15);
    check(JSON.stringify(await values(page.locator(".ff-value-output .ff-number"))) === JSON.stringify(beforeSourceChange), "inspecting another source does not remove terms or change the head output");
    const valueTerm = await values(page.locator(".ff-value-term .ff-number"));
    check(valueTerm[0] === layer.operations.attention[7][12][94] && valueTerm[1] === layer.operations.value[7][94][15] && valueTerm[2] === valueTerm[0] * valueTerm[1], "one selected source contribution uses the recorded A and V");
    await change(page, "head", 0);
    await checkValueDiagram(page, layer, 12, 0, 15);
    await change(page, "reaction", 27);
    await checkValueDiagram(page, layer, 27, 0, 15);
    await change(page, "reaction", 12);
    await change(page, "head", 7);
    await change(page, "layer", 2);
    await checkValueDiagram(page, await json(sample(870).layers[2]), 12, 7, 15);
    await change(page, "layer", 1);
    await page.locator('[data-key="vector"] > summary').click();
    check(await page.locator('[data-key="vector"] tbody tr').count() === 16, "expanded head output shows all 16 features, not four heads");
    await page.locator(".ff-value-matrix").screenshot({ path: `${screenshots}/${engine}-value-matrix.png` });
    await page.locator("#ff-feature").fill("127");
    await page.locator("#ff-feature").press("Tab");
    await busy(page);
    check((await values(page.locator(".ff-residual .ff-number")))[2] === layer.operations.embedding_residual[12][127], "embedding feature 127 is inspectable");

    await step(page, 3);
    check((await values(page.locator(".ff-residual .ff-number")))[2] === layer.operations.output_c[12][0], "feed-forward residual uses exact output");
    await page.locator('[data-key="hidden"] > summary').click();
    await page.locator("#ff-hidden").fill("639");
    await page.locator("#ff-hidden").press("Tab");
    await busy(page);
    const hiddenValues = await values(page.locator('[data-key="hidden"] td .ff-number'));
    check(hiddenValues[1] === layer.operations.linear1[12][639] && hiddenValues[2] === layer.operations.gelu[12][639], "last hidden feature uses exact linear/GELU values");
    await page.locator("#ff-hidden").fill("1000");
    await page.locator("#ff-hidden").press("Tab");
    check(await page.locator("#ff-hidden").inputValue() === "639", "invalid feature indices do not alter data");
    await page.screenshot({ path: `${screenshots}/${engine}-feedforward.png`, fullPage: true });

    await step(page, 4);
    const overview870 = await json(sample(870).overview);
    check(JSON.stringify(await values(page.locator(".ff-state-list .ff-number"))) === JSON.stringify([overview870.initial_c[12], ...overview870.states.map(state => state[12])]), "all four trajectory states are exact");
    await page.goBack();
    await busy(page);
    check((await page.locator("#ff-step-label").textContent()).includes("Step 4"), "browser Back restores previous stage");
    await step(page, 5);
    const resultValues = await values(page.locator(".ff-result-values .ff-number"));
    check(resultValues[0] === overview870.states[2][12] && resultValues[1] === overview870.targets[12], "final prediction and matching target are exact");
    check(await page.locator("#ff-next").isDisabled(), "last stage has no dead next action");
    await page.locator('[data-key="results"] > summary').click();
    await page.locator("#ff-search").fill("EX_o2_e");
    await busy(page);
    check(await page.locator(".ff-reaction-link").count() === 1, "search filters all reactions");
    await page.locator(".ff-reaction-link").click();
    await busy(page);
    check(await page.locator("#ff-reaction").inputValue() === "35", "result selection updates contextual reaction");
    await page.locator("#ff-search").fill("");
    await busy(page);
    check(await page.locator(".ff-reaction-link").count() === 6, "result list is paginated");
    await page.locator('[data-page="1"]').click();
    await busy(page);
    check((await page.locator(".ff-pagination").innerText()).includes("2 / 16"), "complete reaction set is available in pages");
    await page.screenshot({ path: `${screenshots}/${engine}-results.png`, fullPage: true });

    for (const width of [1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      for (let stage = 0; stage < 6; stage++) {
        await step(page, stage);
        await fits(page, `${engine}, ${width}px, stage ${stage}`);
        if (stage === 0) {
          check(await page.locator(".ff-vector-full .ff-feature").count() === 128, `all 128 features remain visible at ${width}px`);
          check(await page.locator(".ff-vector-full").evaluate(chart => chart.scrollWidth <= chart.clientWidth), `full feature chart fits at ${width}px`);
          check(await page.locator(".ff-initial-matrices").evaluate(chart => chart.scrollWidth <= chart.clientWidth), `X/c matrix overview fits at ${width}px`);
          check(await page.locator(".ff-initial-c-cell").count() === 95, `all 95 scalar rows remain represented at ${width}px`);
        }
        if (stage === 1) {
          check(await page.locator(".ff-matrix-product").evaluate(chart => chart.scrollWidth <= chart.clientWidth), `linked matrices fit at ${width}px`);
          check(await page.locator(".ff-weight-cell").count() === 95, `all 95 softmax weights remain represented at ${width}px`);
        }
        if (stage === 2) {
          check(await page.locator(".ff-value-matrix .ff-matrix-product").evaluate(chart => chart.scrollWidth <= chart.clientWidth), `A × V matrices fit at ${width}px`);
          check(await page.locator(".ff-value-cell").count() === 285, `all three 95-source strips remain complete at ${width}px`);
        }
      }
    }
    await page.locator("#theme-select").selectOption("light");
    check(await page.locator("html").getAttribute("data-theme") === "light", "existing light theme applies");
    await step(page, 0);
    const lightColors = await page.locator(".ff-vector-full").evaluate(chart => ["negative", "positive"].map(sign => getComputedStyle(chart.querySelector(`.${sign} i`)).backgroundColor.match(/\d+/g).map(Number)));
    check(lightColors[0][2] > lightColors[0][0] && lightColors[1][0] > lightColors[1][2], "blue/red signs remain distinct in light mode");
    await page.screenshot({ path: `${screenshots}/${engine}-mobile-vector-light.png`, fullPage: true });
    await step(page, 1);
    await page.screenshot({ path: `${screenshots}/${engine}-mobile-light.png`, fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.screenshot({ path: `${screenshots}/${engine}-desktop-light.png`, fullPage: true });
    await context.close();
    console.log(`${engine}: navigation, numerical displays, controls, focus, history, pagination, themes, and 320–1440px layouts passed`);
  } finally { await browser.close(); }
}

// Loading failures, retry, race conditions, progressive loading, and no-JS fallback.
const browser = await driver.chromium.launch({ headless: true });
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(error.message));
  let failManifest = true, failLayer = true;
  await page.route("**/data/fluxformer/manifest.json", route => failManifest ? route.fulfill({ status: 503, body: "Unavailable" }) : route.continue());
  await page.goto(`${base}/fluxformer/`);
  await page.waitForSelector("#ff-retry:not([hidden])");
  check((await page.locator("#ff-status").textContent()).includes("503"), "manifest failure is explained");
  failManifest = false;
  await page.locator("#ff-retry").click();
  await page.waitForSelector(".ff-initial-matrices");
  await page.route("**/sample-000722-layer-0.json.gz", route => failLayer ? route.fulfill({ status: 503, body: "Unavailable" }) : route.continue());
  await step(page, 1);
  check((await page.locator(".ff-notice").innerText()).includes("503"), "layer failure is explained");
  failLayer = false;
  await page.locator("[data-reload]").click();
  await page.waitForSelector(".ff-attention-row");
  check(await page.locator(".ff-attention-row").count() === 5, "retry loads the real layer");
  let delayedResponse;
  await page.route("**/sample-000870-layer-0.json.gz", async route => {
    const response = await route.fetch();
    await new Promise(resolve => setTimeout(resolve, 350));
    await route.fulfill({ response });
    delayedResponse = true;
  });
  const pending = page.waitForRequest("**/sample-000870-layer-0.json.gz");
  await page.locator("#ff-sample").selectOption("870");
  await pending;
  await change(page, "sample", 722);
  while (!delayedResponse) await new Promise(resolve => setTimeout(resolve, 50));
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  check((await page.locator("#ff-condition-note").textContent()).startsWith("Row 722"), "late old-sample response cannot replace the current condition");
  await context.close();

  const limited = await browser.newContext();
  await limited.addInitScript(() => { globalThis.DecompressionStream = undefined; });
  const limitedPage = await limited.newPage();
  await limitedPage.goto(`${base}/fluxformer/#predictions`);
  await limitedPage.waitForSelector(".ff-result-values");
  await step(limitedPage, 1);
  check((await limitedPage.locator(".ff-notice").innerText()).includes("decompression support"), "older browsers retain a useful fallback");
  await step(limitedPage, 5);
  check(await limitedPage.locator(".ff-result-values").count() === 1, "final outputs remain available without decompression");
  await limited.close();

  const noJS = await browser.newContext({ javaScriptEnabled: false });
  const staticPage = await noJS.newPage();
  await staticPage.goto(`${base}/fluxformer/`);
  check((await staticPage.locator("noscript").innerText()).includes("Enable JavaScript"), "no-JS explanation and data link");
  await noJS.close();
} finally { await browser.close(); }
check(errors.length === 0, `no uncaught browser errors: ${errors.join("; ")}`);
console.log(`PASS: ${assertions} browser assertions. Screenshots: ${screenshots}`);
