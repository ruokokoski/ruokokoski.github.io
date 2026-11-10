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

for (const engine of ["chromium", "webkit"]) {
  const browser = await driver[engine].launch({ headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(`${engine}: ${error.message}`));
    const requests = [];
    page.on("request", request => requests.push(request.url()));
    await page.goto(`${base}/fluxformer/`);
    await page.waitForSelector(".ff-token");
    check((await page.locator(".site-nav a").allTextContents()).join("|") === "Home|FluxFormer|Research|Blog", "navigation order");
    check(await page.locator(".site-nav a[aria-current=page]").textContent() === "FluxFormer", "active navigation");
    check(!requests.some(url => url.endsWith(".json.gz")), "initialization does not download full layers");
    check(await page.locator(".ff-token").count() === 6, "only six representative tokens visible");
    check(await page.locator("#ff-layer-field").isHidden(), "layer control is contextual");
    await page.locator('[data-key="nutrients"] > summary').click();
    await change(page, "nutrient", 18);
    const overview722 = await json(sample(722).overview);
    check((await values(page.locator('[data-key="nutrients"] .ff-number')))[0] === overview722.constraints[18], "displayed oxygen bound is exact");
    const fixedExamples = await page.locator(".ff-token").evaluateAll(tokens => tokens.map(token => token.dataset.reaction));
    await change(page, "reaction", 0);
    check(JSON.stringify(await page.locator(".ff-token").evaluateAll(tokens => tokens.map(token => token.dataset.reaction))) === JSON.stringify(fixedExamples), "choosing another reaction preserves all six fixed examples");
    check(await page.locator(".ff-inspection code").textContent() === "ACALD", "selected reaction appears in its own panel");
    check(await page.locator(".ff-token.is-selected").count() === 0, "unrelated examples are not highlighted");
    await change(page, "reaction", 25);
    check((await values(page.locator(".ff-initial-scalar .ff-number")))[0] === overview722.initial_c[25], "separate panel displays the selected input scalar exactly");
    await page.locator('[data-key="embedding"] > summary').click();
    check((await values(page.locator('[data-key="embedding"] .ff-equation .ff-number')))[0] === meta.initial_x[25][0], "embedding inspector follows the separate selection");
    await page.locator('.ff-token[data-reaction="27"]').click();
    await busy(page);
    check(await page.locator("#ff-reaction").inputValue() === "27" && await page.locator(".ff-inspection code").textContent() === "EX_glc__D_e", "clicking a fixed example updates the selector and inspection panel");
    check(await page.locator('.ff-token[data-reaction="27"]').getAttribute("aria-pressed") === "true", "selected example remains highlighted");
    await change(page, "reaction", 12);
    await page.screenshot({ path: `${screenshots}/${engine}-initial.png`, fullPage: true });

    await step(page, 1);
    check(await page.locator(".ff-attention-row").count() === 5, "attention shows five sources");
    const firstLayer = await json(sample(722).layers[0]);
    const firstSource = Number(await page.locator(".ff-attention-row.is-selected").getAttribute("data-source"));
    check((await values(page.locator(".ff-equation .ff-number")))[0] === firstLayer.operations.scaled_logits[0][12][firstSource], "selected logit is exact");
    const firstWeights = await page.locator(".ff-attention-row strong").allTextContents();
    await page.locator(".ff-attention-row").nth(1).focus();
    await page.keyboard.press("Enter");
    await busy(page);
    check(await page.evaluate(() => document.activeElement.matches(".ff-attention-row.is-selected")), "source selection preserves keyboard focus");
    await page.locator('[data-key="attention"] > summary').click();
    await change(page, "source", 94);
    check((await values(page.locator(".ff-equation .ff-number")))[0] === firstLayer.operations.scaled_logits[0][12][94], "any of 95 sources can be inspected");
    check(await page.locator('[data-key="attention"]').getAttribute("open") !== null, "details remain open after local selection");
    await change(page, "sample", 870);
    check(JSON.stringify(await page.locator(".ff-attention-row strong").allTextContents()) === JSON.stringify(firstWeights), "first-layer attention display stays fixed across conditions");
    await change(page, "head", 7);
    await change(page, "layer", 1);
    check((await page.locator(".ff-insight").innerText()).includes("Later attention"), "later-layer context explanation");
    await page.screenshot({ path: `${screenshots}/${engine}-attention.png`, fullPage: true });

    await step(page, 2);
    const layer = await json(sample(870).layers[1]);
    const scalarNumbers = await values(page.locator(".ff-residual .ff-number"));
    check(scalarNumbers[0] === layer.input_c[12][0] && scalarNumbers[2] === layer.operations.scalar_residual[12][0], "scalar residual uses exact recorded values");
    await page.locator('[data-channel="embedding"]').click();
    await busy(page);
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
      }
    }
    await page.locator("#theme-select").selectOption("light");
    check(await page.locator("html").getAttribute("data-theme") === "light", "existing light theme applies");
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
  await page.waitForSelector(".ff-token");
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
