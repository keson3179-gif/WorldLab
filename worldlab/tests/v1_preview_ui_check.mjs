import { spawn, execFile } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const demo = path.join(root, "v1_demo");
const shots = path.join(demo, "acceptance_screenshots");
const failures = [];

function fail(message) {
  failures.push(message);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close((error) => (error ? reject(error) : resolve(port)));
    });
  });
}

function contentType(file) {
  if (file.endsWith(".html")) return "text/html; charset=utf-8";
  if (file.endsWith(".css")) return "text/css; charset=utf-8";
  if (file.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (file.endsWith(".json")) return "application/json; charset=utf-8";
  return "application/octet-stream";
}

function readExport(relative) {
  return fs.readFileSync(path.join(demo, "export", ...relative.split("/")));
}

function startFixtureServer() {
  const hits = new Map();
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    const pathname = decodeURIComponent(url.pathname);
    response.setHeader("cache-control", "no-store");
    const send = (status, body, type = "application/json; charset=utf-8", delay = 0) => {
      setTimeout(() => {
        response.writeHead(status, { "content-type": type });
        response.end(body);
      }, delay);
    };
    if (pathname.startsWith("/static/")) {
      const file = path.resolve(demo, ...pathname.split("/").filter(Boolean));
      if (!file.startsWith(path.resolve(demo) + path.sep) || !fs.existsSync(file)) {
        send(404, "missing", "text/plain");
        return;
      }
      send(200, fs.readFileSync(file), contentType(file));
      return;
    }
    const fixture = pathname.match(/^\/fixture\/([^/]+)\/(.+)$/);
    if (fixture) {
      const [, mode, relative] = fixture;
      const key = mode + ":" + relative;
      hits.set(key, (hits.get(key) || 0) + 1);
      const count = hits.get(key);
      if (mode === "slow") {
        send(200, readExport(relative), contentType(relative), 700);
        return;
      }
      if (mode === "fail-catalog" && relative === "catalog.json") {
        send(500, "catalog failed", "text/plain");
        return;
      }
      if (mode === "bad-json" && relative === "catalog.json") {
        send(200, "{not json", "application/json; charset=utf-8");
        return;
      }
      if (mode === "incomplete" && relative === "catalog.json") {
        send(200, JSON.stringify({ seeds: [1700] }));
        return;
      }
      if (mode === "fail-replay" && relative === "replays/1700.json") {
        send(404, "missing replay", "text/plain");
        return;
      }
      if (mode === "empty-replay" && relative === "replays/1700.json") {
        send(200, JSON.stringify({ seed: 1700, worlds: [] }));
        return;
      }
      if (mode === "retry" && relative === "catalog.json" && count === 1) {
        send(503, "try again", "text/plain");
        return;
      }
      if (mode === "race" && relative === "catalog.json" && count === 1) {
        send(500, "stale failure", "text/plain", 900);
        return;
      }
      try {
        send(200, readExport(relative), contentType(relative));
      } catch {
        send(404, "missing", "text/plain");
      }
      return;
    }
    send(404, "not found", "text/plain");
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port, hits }));
  });
}

function pngSize(buffer) {
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

function chromePath() {
  const candidates = [
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  ];
  return candidates.find((candidate) => fs.existsSync(candidate));
}

async function waitJson(url) {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response.json();
    } catch {
      // Chrome is still starting.
    }
    await sleep(100);
  }
  throw new Error("browser debugging port did not open");
}

function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    let sequence = 0;
    const pending = new Map();
    const listeners = new Map();
    ws.addEventListener("open", () => {
      resolve({
        send(method, params = {}) {
          const id = ++sequence;
          return new Promise((done, rejectSend) => {
            pending.set(id, { done, rejectSend });
            ws.send(JSON.stringify({ id, method, params }));
          });
        },
        on(method, fn) {
          listeners.set(method, [...(listeners.get(method) || []), fn]);
        },
        close() {
          ws.close();
        },
      });
    });
    ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const { done, rejectSend } = pending.get(message.id);
        pending.delete(message.id);
        if (message.error) rejectSend(new Error(JSON.stringify(message.error)));
        else done(message.result || {});
      } else if (message.method) {
        for (const fn of listeners.get(message.method) || []) fn(message.params || {});
      }
    });
    ws.addEventListener("error", () => reject(new Error("devtools websocket failed")));
  });
}

async function openBrowser(port) {
  const version = await waitJson(`http://127.0.0.1:${port}/json/version`);
  const browser = await connect(version.webSocketDebuggerUrl);
  const created = await browser.send("Target.createTarget", { url: "about:blank" });
  const targets = await waitJson(`http://127.0.0.1:${port}/json/list`);
  const page = targets.find((item) => item.id === created.targetId) || targets.find((item) => item.type === "page");
  const session = await connect(page.webSocketDebuggerUrl);
  await session.send("Page.enable");
  await session.send("Runtime.enable");
  await session.send("Page.addScriptToEvaluateOnNewDocument", {
    source: `(() => {
      window.__errors = [];
      window.__intervals = 0;
      window.addEventListener("error", (event) => window.__errors.push(String(event.message)));
      window.addEventListener("unhandledrejection", (event) => window.__errors.push(String(event.reason && event.reason.message || event.reason)));
      const originalSet = window.setInterval;
      const originalClear = window.clearInterval;
      window.setInterval = function () {
        window.__intervals += 1;
        return originalSet.apply(this, arguments);
      };
      window.clearInterval = function (id) {
        window.__intervals = Math.max(0, window.__intervals - 1);
        return originalClear.call(this, id);
      };
    })();`,
  });
  return { browser, session };
}

async function evaluate(session, expression) {
  const result = await session.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) {
    throw new Error(JSON.stringify(result.exceptionDetails));
  }
  return result.result ? result.result.value : undefined;
}

async function navigate(session, url) {
  const loaded = new Promise((resolve) => session.on("Page.loadEventFired", resolve));
  await session.send("Page.navigate", { url });
  await loaded;
  await sleep(50);
}

async function setViewport(session, width, height) {
  await session.send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await sleep(50);
}

async function waitFor(session, expression, timeout = 8000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    if (await evaluate(session, expression)) return true;
    await sleep(50);
  }
  return false;
}

function check(condition, message) {
  if (!condition) fail(message);
}

async function capture(session, file, cssWidth, cssHeight, y = 0) {
  const image = await session.send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
    clip: { x: 0, y, width: cssWidth, height: cssHeight, scale: 1 },
  });
  const buffer = Buffer.from(image.data, "base64");
  fs.mkdirSync(shots, { recursive: true });
  fs.writeFileSync(file, buffer);
  return pngSize(buffer);
}

async function captureComplete(session, name, cssWidth) {
  const content = await evaluate(session, `({width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight})`);
  const full = await capture(session, path.join(shots, name + ".png"), cssWidth, content.height);
  const record = { name, cssViewportWidth: cssWidth, contentWidth: content.width, contentHeight: content.height, files: [] };
  if (full.width === cssWidth && full.height === content.height) {
    record.files.push({ file: name + ".png", png: full, covers: "y 0 to " + content.height });
    return record;
  }
  fs.rmSync(path.join(shots, name + ".png"), { force: true });
  const sliceHeight = Math.max(full.height, 1);
  if (sliceHeight >= content.height && full.width === cssWidth) {
    record.files.push({ file: name + ".png", png: full, covers: "y 0 to " + content.height });
    return record;
  }
  let y = 0;
  let index = 1;
  while (y < content.height) {
    const height = Math.min(sliceHeight, content.height - y);
    const file = `${name}_y${y}.png`;
    const size = await capture(session, path.join(shots, file), cssWidth, height, y);
    record.files.push({ file, png: size, covers: "y " + y + " to " + (y + height), clip: { x: 0, y, width: cssWidth, height } });
    check(size.width === cssWidth && size.height === height, name + " segment " + file + " was cropped");
    y += height;
    index += 1;
    if (index > 12) {
      fail(name + " screenshot segmentation did not cover the page");
      break;
    }
  }
  return record;
}

async function pageState(session) {
  return evaluate(session, `(() => {
    const ids = ["memory","transform","q-label","seed","left-schedule","right-schedule","play","pause","restart","back","forward","speed","time-slider","jump-0","jump-50","jump-51","jump-500"];
    const controls = Object.fromEntries(ids.map((id) => [id, document.getElementById(id).disabled]));
    const plain = (rect) => ({ left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom, width: rect.width, height: rect.height });
    const left = document.getElementById("left-panel");
    const right = document.getElementById("right-panel");
    const ruler = plain(document.querySelector(".ruler").getBoundingClientRect());
    const slider = plain(document.getElementById("time-slider").getBoundingClientRect());
    const jumps = [...document.querySelectorAll(".jumps button")].map((node) => plain(node.getBoundingClientRect()));
    const tick = (selector) => {
      const box = document.querySelector(selector).getBoundingClientRect();
      return { left: box.left, right: box.right };
    };
    const places = (panel) => [...panel.querySelectorAll(".place")].map((place) => {
      const box = plain(place.getBoundingClientRect());
      const agents = [...place.querySelectorAll(".agent")].map((agent) => plain(agent.getBoundingClientRect()));
      return {
        title: place.querySelector("h3").textContent,
        text: place.innerText,
        box,
        overflow: getComputedStyle(place).overflow,
        agents,
      };
    });
    return {
      phase: document.body.dataset.phase,
      status: document.getElementById("load-status").textContent,
      retryVisible: !document.getElementById("retry").hidden && document.getElementById("retry").getClientRects().length > 0,
      controls,
      errors: window.__errors,
      intervals: window.__intervals,
      clock: document.getElementById("clock").textContent,
      caption: document.getElementById("time-caption").textContent,
      note: document.getElementById("switch-note").textContent,
      slider: document.getElementById("time-slider").value,
      leftTitle: left.querySelector("h2") ? left.querySelector("h2").textContent : "",
      rightTitle: right.querySelector("h2") ? right.querySelector("h2").textContent : "",
      leftText: left.innerText,
      rightText: right.innerText,
      hash: document.getElementById("active-hash").textContent,
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      leftBox: plain(left.getBoundingClientRect()),
      rightBox: plain(right.getBoundingClientRect()),
      leftPlaces: places(left),
      rightPlaces: places(right),
      ruler,
      sliderBox: slider,
      ticks: { t0: tick(".tick-0"), t50: tick(".tick-50"), t51: tick(".tick-51"), t500: tick(".tick-500") },
      jumps,
      stagesColumns: getComputedStyle(document.querySelector(".stages")).gridTemplateColumns,
    };
  })()`);
}

function overlaps(a, b) {
  return a.left < b.right - 1 && a.right > b.left + 1 && a.top < b.bottom - 1 && a.bottom > b.top + 1;
}

function assertLayout(state, width, mode) {
  check(state.scrollWidth <= state.clientWidth + 1, width + " has horizontal overflow " + state.scrollWidth + "/" + state.clientWidth);
  const sideBySide = Math.abs(state.leftBox.top - state.rightBox.top) <= 2 && state.leftBox.right <= state.rightBox.left + 2 && state.rightBox.left > state.leftBox.left + 100;
  const stacked = Math.abs(state.leftBox.left - state.rightBox.left) <= 2 && state.leftBox.bottom <= state.rightBox.top + 2;
  if (mode === "side") check(sideBySide, width + " panels are not side by side");
  if (mode === "stack") check(stacked, width + " panels are not stacked");
  check(!overlaps(state.leftBox, state.rightBox), width + " panels overlap");
  for (const side of ["leftPlaces", "rightPlaces"]) {
    check(state[side].length === 5, width + " " + side + " does not show five places");
    const titles = state[side].map((place) => place.title);
    check(titles.join("|") === state.leftPlaces.map((place) => place.title).join("|"), width + " place order differs");
    state[side].forEach((place, index) => {
      check(place.overflow !== "hidden", width + " place clips overflow");
      const agents = place.agents;
      agents.forEach((agent) => {
        check(agent.width >= 24 && agent.height >= 16, width + " agent chip is unreadable");
        check(agent.left >= place.box.left - 1 && agent.right <= place.box.right + 1 && agent.top >= place.box.top - 1 && agent.bottom <= place.box.bottom + 1, width + " agent is outside its place card");
      });
      for (let other = index + 1; other < state[side].length; other += 1) {
        check(!overlaps(place.box, state[side][other].box), width + " place cards overlap");
      }
    });
    check(state[side].reduce((sum, place) => sum + place.agents.length, 0) === 20, width + " does not show 20 agents");
  }
}

async function selectValue(session, id, value) {
  await evaluate(session, `(() => { const node = document.getElementById(${JSON.stringify(id)}); node.value = ${JSON.stringify(value)}; node.dispatchEvent(new Event("change", {bubbles: true})); })()`);
}

async function main() {
  const executable = chromePath();
  if (!executable) throw new Error("Chrome or Edge was not found");
  const { server, port } = await startFixtureServer();
  const debugPort = await freePort();
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "worldlab-v1-"));
  const chrome = spawn(executable, [
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--disable-extensions",
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profile}`,
    "about:blank",
  ], { stdio: "ignore" });
  const records = [];
  try {
    const { browser, session } = await openBrowser(debugPort);
    const base = `http://127.0.0.1:${port}`;

    await setViewport(session, 1440, 900);
    await navigate(session, `${base}/static/index.html?data=/fixture/slow`);
    let state = await pageState(session);
    check(state.phase === "loading", "slow load was not still loading");
    check(state.status.includes("正在載入既有回放資料"), "loading message missing");
    check(Object.values(state.controls).every(Boolean), "controls were enabled before data loaded");
    check(state.intervals === 0, "a timer started before data loaded");
    await evaluate(session, `["play","jump-50","forward","time-slider"].forEach((id) => document.getElementById(id).dispatchEvent(new MouseEvent("click", {bubbles:true})))`);
    state = await pageState(session);
    check(state.errors.length === 0, "early clicks raised " + state.errors.join("; "));
    check(state.phase === "loading" && state.intervals === 0 && state.clock.includes("—"), "early clicks changed playback");
    check(await waitFor(session, `document.body.dataset.phase === "ready"`), "slow load did not become ready");
    state = await pageState(session);
    check(Object.values(state.controls).every((disabled) => disabled === false), "controls stayed disabled after ready");
    check(state.leftTitle.startsWith("左視窗") && state.rightTitle.startsWith("右視窗"), "panel names missing");
    check(state.leftTitle.includes("CUM_LOGMAX__q25") && state.rightTitle.includes("EO"), "default condition missing");
    const lateLine = (text) => {
      const match = text.match(/第 101–500 步的累積訪問集中度：[^\n]+/);
      return match ? match[0] : "";
    };
    const savedLate = { left: lateLine(state.leftText), right: lateLine(state.rightText) };
    await evaluate(session, `document.getElementById("play").click(); document.getElementById("play").click();`);
    await sleep(700);
    state = await pageState(session);
    const played = Number(state.slider);
    check(state.intervals === 1, "repeated play created " + state.intervals + " timers");
    check(played >= 1 && played <= 3, "playback advanced at the wrong speed: " + played);
    await evaluate(session, `document.getElementById("jump-50").click()`);
    await sleep(500);
    state = await pageState(session);
    check(state.slider === "50" && state.clock === "committed t = 50" && state.intervals === 0, "jump did not pause at t=50");
    check(state.caption.includes("切換前最後一個 committed state"), "t=50 caption is wrong");
    for (const target of [0, 50, 51, 500]) {
      await evaluate(session, `document.getElementById("jump-${target}").click()`);
      state = await pageState(session);
      check(state.slider === String(target) && state.clock === "committed t = " + target, "jump " + target + " did not sync clock and slider");
      check(lateLine(state.leftText) === savedLate.left && lateLine(state.rightText) === savedLate.right, "jump " + target + " changed late HHI");
    }
    const hhiAtEnd = state.leftText.match(/累積訪問集中度：([^\n；]+)/);
    await evaluate(session, `document.getElementById("jump-0").click()`);
    const hhiAtStart = (await pageState(session)).leftText.match(/累積訪問集中度：([^\n；]+)/);
    check(hhiAtEnd && hhiAtStart && hhiAtEnd[1] === hhiAtStart[1], "late HHI changed between t=500 and t=0");
    await evaluate(session, `document.getElementById("jump-51").click()`);
    state = await pageState(session);
    check(state.caption.includes("只有 EO 與 OE"), "t=51 caption treats every schedule as switching");
    check(state.note.includes("右視窗 EO") && !state.note.includes("都不會"), "EO switch note missing");
    check(state.leftText.includes("當下最大場所人數：20") && state.rightText.includes("當下最大場所人數：13"), "t=51 occupancy was not compared");
    await evaluate(session, `document.getElementById("jump-50").click()`);
    state = await pageState(session);
    const crowded = state.leftPlaces.some((place) => place.agents.length === 20);
    check(crowded, "twenty agents were not together in one place at t=50");
    assertLayout(state, 1440, "side");
    check(Math.abs(state.ticks.t0.left - state.ruler.left) <= 2, "t=0 mark is not at the slider start");
    check(Math.abs(state.ticks.t50.left - (state.ruler.left + state.ruler.width * 0.1)) <= 2, "t=50 mark is not at 10%");
    check(Math.abs(state.ticks.t51.left - (state.ruler.left + state.ruler.width * 0.102)) <= 2, "t=51 mark is not at 10.2%");
    check(Math.abs(state.ticks.t500.right - state.ruler.right) <= 2, "t=500 mark is not at the slider end");
    check(Math.abs(state.ruler.left - state.sliderBox.left) <= 2 && Math.abs(state.ruler.right - state.sliderBox.right) <= 2, "ruler is not aligned to the slider");
    state.jumps.slice(1).forEach((button, index) => {
      if (Math.abs(button.top - state.jumps[index].top) <= 2) {
        check(button.left - state.jumps[index].right < 24, "jump buttons are spread like a scale");
      }
    });
    await setViewport(session, 1440, (await evaluate(session, "document.documentElement.scrollHeight")));
    records.push(await captureComplete(session, "desktop_1440", 1440));

    await setViewport(session, 1024, 900);
    state = await pageState(session);
    assertLayout(state, 1024, "side");
    await setViewport(session, 1024, (await evaluate(session, "document.documentElement.scrollHeight")));
    records.push(await captureComplete(session, "desktop_1024", 1024));

    await setViewport(session, 768, 900);
    state = await pageState(session);
    assertLayout(state, 768, "stack");
    await setViewport(session, 768, (await evaluate(session, "document.documentElement.scrollHeight")));
    records.push(await captureComplete(session, "narrow_768", 768));

    await setViewport(session, 1440, 900);
    const before = await pageState(session);
    await selectValue(session, "q-label", "q05");
    state = await pageState(session);
    check(state.leftTitle.includes("q05") && state.rightTitle.includes("q05"), "q change did not update both titles");
    check(state.hash !== before.hash, "q change did not update trajectory hashes");
    check(state.leftText !== before.leftText && state.rightText !== before.rightText, "q change did not update metrics");
    check(state.leftText.includes("未形成"), "q05 did not show the absent event");
    await selectValue(session, "memory", "RW50");
    await selectValue(session, "transform", "ZRAW");
    state = await pageState(session);
    check(state.leftTitle.includes("RW50_ZRAW__q05") && state.rightTitle.includes("RW50_ZRAW__q05"), "memory and transform did not update both titles");
    const beforeSeed = state.hash;
    await selectValue(session, "seed", "2699");
    state = await pageState(session);
    check(state.hash !== beforeSeed && state.leftTitle.includes("RW50_ZRAW__q05"), "seed change did not update the trajectory");
    await selectValue(session, "memory", "CUM");
    await selectValue(session, "transform", "LOGMAX");
    await selectValue(session, "q-label", "q25");
    await selectValue(session, "seed", "1700");
    await selectValue(session, "right-schedule", "OE");
    state = await pageState(session);
    check(state.rightTitle.includes("OE") && state.rightText.includes("不適用／沒有合格樣本"), "OE maintenance was not shown as inapplicable");
    await selectValue(session, "left-schedule", "EE");
    await selectValue(session, "right-schedule", "OO");
    await evaluate(session, `document.getElementById("jump-51").click()`);
    state = await pageState(session);
    check(state.note.includes("都不會在 t=51 更換來源"), "EE/OO was described as switching source");
    await selectValue(session, "right-schedule", "S0");
    state = await pageState(session);
    check(state.note.includes("都不會") && state.rightTitle.includes("S0"), "S0 was described as switching source");

    async function expectError(mode) {
      await navigate(session, `${base}/static/index.html?data=/fixture/${mode}`);
      check(await waitFor(session, `document.body.dataset.phase === "error"`), mode + " did not show an error");
      state = await pageState(session);
      check(state.status.includes("載入既有回放資料失敗"), mode + " error text missing");
      check(state.retryVisible, mode + " retry button missing");
      check(Object.values(state.controls).every(Boolean), mode + " left controls enabled");
      check(state.intervals === 0 && state.leftTitle === "", mode + " showed a playable replay");
    }
    await setViewport(session, 1024, 900);
    await expectError("fail-catalog");
    await setViewport(session, 1024, (await evaluate(session, "document.documentElement.scrollHeight")));
    records.push(await captureComplete(session, "load_error", 1024));
    await setViewport(session, 1024, 900);
    await expectError("fail-replay");
    await expectError("bad-json");
    await expectError("incomplete");
    await expectError("empty-replay");

    await navigate(session, `${base}/static/index.html?data=/fixture/retry`);
    check(await waitFor(session, `document.body.dataset.phase === "error"`), "retry fixture did not fail first");
    await evaluate(session, `document.getElementById("retry").click()`);
    check(await waitFor(session, `document.body.dataset.phase === "ready"`), "retry did not recover");
    state = await pageState(session);
    check(Object.values(state.controls).every((disabled) => disabled === false), "retry left controls disabled");
    await evaluate(session, `document.getElementById("play").click(); document.getElementById("play").click();`);
    await sleep(600);
    state = await pageState(session);
    check(state.intervals === 1, "retry created multiple timers: " + state.intervals);
    await evaluate(session, `document.getElementById("pause").click()`);

    await navigate(session, `${base}/static/index.html?data=/fixture/race`);
    await sleep(200);
    await evaluate(session, `document.getElementById("retry").click()`);
    check(await waitFor(session, `document.body.dataset.phase === "ready"`, 4000), "newer load did not become ready");
    await sleep(1200);
    state = await pageState(session);
    check(state.phase === "ready" && state.status.includes("已載入"), "stale failure overwrote the successful retry");
    check(state.errors.length === 0, "page errors: " + state.errors.join("; "));

    session.close();
    browser.close();
  } catch (error) {
    fail(error && error.stack ? error.stack : String(error));
  } finally {
    server.close();
    if (chrome.pid) {
      await new Promise((resolve) => execFile("taskkill", ["/PID", String(chrome.pid), "/T", "/F"], () => resolve()));
    }
    await sleep(300);
    try {
      fs.rmSync(profile, { recursive: true, force: true });
    } catch {
      // Chrome can keep the temporary profile locked after shutdown.
    }
  }
  fs.mkdirSync(shots, { recursive: true });
  fs.writeFileSync(path.join(shots, "screenshot_record.json"), JSON.stringify({ screenshots: records, failures }, null, 2));
  if (failures.length) {
    console.error(failures.join("\n"));
    process.exit(1);
  }
  console.log(JSON.stringify({ ok: true, screenshots: records.map((item) => item.name) }));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
