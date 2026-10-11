const COLORS = ["#9a3412", "#1d4ed8", "#047857", "#7c3aed", "#b45309", "#0f766e", "#be123c", "#1e3a8a", "#4d7c0f", "#6d28d9", "#c2410c", "#0369a1", "#9f1239", "#3f6212", "#4338ca", "#a16207", "#155e75", "#7f1d1d", "#1e40af", "#0e7490"];
const CONTROL_IDS = ["memory", "transform", "q-label", "seed", "left-schedule", "right-schedule", "play", "pause", "restart", "back", "forward", "speed", "time-slider", "jump-0", "jump-50", "jump-51", "jump-500"];

const state = {
  phase: "loading",
  catalog: null,
  replays: {},
  memory: "CUM",
  transform: "LOGMAX",
  q: "q25",
  seed: 1700,
  left: "EE",
  right: "EO",
  t: 0,
  speed: 1,
  timer: null,
};

let loadGeneration = 0;
let controlsBound = false;

function dataBase() {
  const selected = new URLSearchParams(window.location.search).get("data");
  return selected || "../export";
}

function conditionId() {
  return state.memory + "_" + state.transform + "__" + state.q;
}

function changesSource(schedule) {
  return schedule === "EO" || schedule === "OE";
}

function worldFor(schedule) {
  const replay = state.replays[state.seed];
  if (!replay || !replay.worlds) {
    return null;
  }
  const condition = schedule === "S0" ? "S0" : conditionId();
  return replay.worlds.find((item) => item.condition === condition && item.schedule === schedule) || null;
}

function phrase(name) {
  return state.catalog.phrases[name];
}

function exitText(record) {
  const maintenance = record.metrics.maintenance;
  if (!maintenance.eligible) {
    return phrase("not_applicable");
  }
  if (maintenance.original_exit === null) {
    return phrase("no_exit_by_horizon");
  }
  return "原場所退出於 committed t=" + maintenance.original_exit;
}

function releaseText(record) {
  const maintenance = record.metrics.maintenance;
  if (!maintenance.eligible) {
    return phrase("not_applicable");
  }
  if (maintenance.release_time === null) {
    return phrase("no_release_by_horizon");
  }
  return "解除集中於 committed t=" + maintenance.release_time;
}

function occupancy(record) {
  const step = record.steps[state.t];
  const counts = {};
  record.places.forEach((place) => {
    counts[place] = 0;
  });
  const agents = record.agents.map((agentId, index) => {
    const place = record.places[Number(step[index]) - 1];
    counts[place] += 1;
    return { agentId, place, index };
  });
  return { counts, agents, maxCount: Math.max(...Object.values(counts)) };
}

function setControlsEnabled(enabled) {
  CONTROL_IDS.forEach((id) => {
    document.getElementById(id).disabled = !enabled;
  });
}

function setPhase(phase, message) {
  state.phase = phase;
  document.body.dataset.phase = phase;
  document.getElementById("load-status").textContent = message;
  document.getElementById("retry").hidden = phase !== "error";
  setControlsEnabled(phase === "ready");
}

function clearDisplayedReplay() {
  ["left-panel", "right-panel"].forEach((id) => {
    const node = document.getElementById(id);
    node.replaceChildren();
    delete node.dataset.trajectory;
  });
  document.getElementById("active-hash").textContent = "";
  document.getElementById("time-caption").textContent = "";
  document.getElementById("switch-note").textContent = "";
  document.getElementById("clock").textContent = "committed t = —";
}

function pause() {
  if (state.timer !== null) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
}

function fillSelect(id, options, selected) {
  const node = document.getElementById(id);
  node.replaceChildren();
  options.forEach((option) => {
    const item = document.createElement("option");
    item.value = option.id;
    item.textContent = option.label;
    node.appendChild(item);
  });
  node.value = selected;
}

function setupControls() {
  const catalog = state.catalog;
  fillSelect("memory", catalog.memories, state.memory);
  fillSelect("transform", catalog.transforms, state.transform);
  fillSelect("q-label", catalog.q_labels, state.q);
  fillSelect("seed", catalog.seeds.map((seed) => ({ id: String(seed), label: "seed " + seed })), String(state.seed));
  fillSelect("left-schedule", catalog.schedules, state.left);
  fillSelect("right-schedule", catalog.schedules, state.right);
  document.getElementById("q-note").textContent = catalog.q_note;
  document.getElementById("banner").textContent = catalog.banner;
  document.getElementById("source-note").textContent = "來源檔案列在 export/build_record.json。本頁載入 " + catalog.seeds.join("、") + "，合計 219 條既有回放。";
}

function bindControls() {
  if (controlsBound) {
    return;
  }
  controlsBound = true;
  document.getElementById("memory").addEventListener("change", (event) => {
    state.memory = event.target.value;
    render();
  });
  document.getElementById("transform").addEventListener("change", (event) => {
    state.transform = event.target.value;
    render();
  });
  document.getElementById("q-label").addEventListener("change", (event) => {
    state.q = event.target.value;
    render();
  });
  document.getElementById("seed").addEventListener("change", (event) => {
    state.seed = Number(event.target.value);
    render();
  });
  document.getElementById("left-schedule").addEventListener("change", (event) => {
    state.left = event.target.value;
    render();
  });
  document.getElementById("right-schedule").addEventListener("change", (event) => {
    state.right = event.target.value;
    render();
  });
  document.getElementById("play").addEventListener("click", play);
  document.getElementById("pause").addEventListener("click", pause);
  document.getElementById("restart").addEventListener("click", () => jumpTo(0));
  document.getElementById("back").addEventListener("click", () => jumpTo(state.t - 1));
  document.getElementById("forward").addEventListener("click", () => jumpTo(state.t + 1));
  document.getElementById("jump-0").addEventListener("click", () => jumpTo(0));
  document.getElementById("jump-50").addEventListener("click", () => jumpTo(50));
  document.getElementById("jump-51").addEventListener("click", () => jumpTo(51));
  document.getElementById("jump-500").addEventListener("click", () => jumpTo(500));
  document.getElementById("speed").addEventListener("change", (event) => {
    state.speed = Number(event.target.value);
  });
  document.getElementById("time-slider").addEventListener("input", (event) => jumpTo(Number(event.target.value)));
  document.getElementById("retry").addEventListener("click", beginLoad);
}

function play() {
  if (state.phase !== "ready" || state.timer !== null) {
    return;
  }
  state.timer = window.setInterval(() => {
    if (state.t >= 500) {
      pause();
      return;
    }
    state.t = Math.min(500, state.t + state.speed);
    document.getElementById("time-slider").value = String(state.t);
    render();
  }, 250);
}

function jumpTo(value) {
  if (state.phase !== "ready") {
    return;
  }
  pause();
  state.t = Math.max(0, Math.min(500, value));
  document.getElementById("time-slider").value = String(state.t);
  render();
}

function timeCaption() {
  const marks = state.catalog.time_marks;
  if (state.t === 0 || state.t === 50) {
    return marks[String(state.t)];
  }
  if (state.t === 51) {
    return "decision t=51 產生 committed state t=51。只有 EO 與 OE 會在這裡更換來源。";
  }
  return "committed t=" + state.t;
}

function switchNote() {
  const changing = [];
  if (changesSource(state.left)) {
    changing.push("左視窗 " + state.left);
  }
  if (changesSource(state.right)) {
    changing.push("右視窗 " + state.right);
  }
  if (changing.length === 0) {
    return "目前左右視窗都不會在 t=51 更換來源。EE、OO 與 S0 的來源全程不變。";
  }
  return "會在 decision t=51 更換來源的是" + changing.join("、") + "。EE、OO 與 S0 不會更換。";
}

function metricCard(title, body) {
  const node = document.createElement("p");
  node.textContent = title + "：" + body;
  return node;
}

function renderPanel(node, schedule, sideLabel) {
  const record = worldFor(schedule);
  const view = occupancy(record);
  node.replaceChildren();
  const title = document.createElement("h2");
  const condition = schedule === "S0" ? "S0 donor 參考回放" : record.condition + " ／ " + record.schedule;
  title.textContent = sideLabel + "：" + condition;
  node.appendChild(title);
  node.dataset.trajectory = record.trajectory_sha256;
  const places = document.createElement("div");
  places.className = "places";
  state.catalog.places.forEach((place) => {
    const box = document.createElement("div");
    box.className = "place";
    const heading = document.createElement("h3");
    heading.textContent = place.label + "（" + place.id + "）";
    const count = document.createElement("div");
    count.className = "count";
    count.textContent = "當下人數 " + view.counts[place.id];
    const agents = document.createElement("div");
    agents.className = "agents";
    view.agents.filter((agent) => agent.place === place.id).forEach((agent) => {
      const chip = document.createElement("span");
      chip.className = "agent";
      chip.style.background = COLORS[agent.index % COLORS.length];
      chip.textContent = agent.agentId.slice(-2);
      chip.title = agent.agentId;
      agents.appendChild(chip);
    });
    box.append(heading, count, agents);
    places.appendChild(box);
  });
  const metrics = document.createElement("div");
  metrics.className = "metrics";
  const event = record.metrics.event18;
  const late = record.metrics.late_hhi;
  metrics.append(
    metricCard("當下最大場所人數", String(view.maxCount) + "（隨時間改變，只供畫面顯示）"),
    metricCard(late.display_zh, String(late.value) + "；這是整條回放的已知摘要"),
    metricCard("這條軌跡的晚期主要場所", late.tie ? "並列，沒有唯一場所" : String(late.late_dominant)),
    metricCard("早期強集中", record.metrics.early_formation.display_zh),
    metricCard("事件起始", event.display.start),
    metricCard("事件確認", event.display.confirm),
    metricCard("維持分析資格", record.metrics.maintenance.display.eligibility),
    metricCard("原場所退出", exitText(record)),
    metricCard("解除集中", releaseText(record))
  );
  node.append(places, metrics);
}

function render() {
  if (state.phase !== "ready") {
    return;
  }
  const left = worldFor(state.left);
  const right = worldFor(state.right);
  if (!left || !right) {
    failLoad("選定的回放不在已載入的資料裡。");
    return;
  }
  document.getElementById("clock").textContent = "committed t = " + state.t;
  document.getElementById("time-slider").value = String(state.t);
  document.getElementById("time-caption").textContent = timeCaption();
  document.getElementById("switch-note").textContent = switchNote();
  renderPanel(document.getElementById("left-panel"), state.left, "左視窗");
  renderPanel(document.getElementById("right-panel"), state.right, "右視窗");
  document.getElementById("active-hash").textContent = "左 " + left.trajectory_sha256 + " ｜ 右 " + right.trajectory_sha256;
}

function failLoad(message) {
  pause();
  state.catalog = null;
  state.replays = {};
  clearDisplayedReplay();
  setPhase("error", message);
}

async function readJson(response, label) {
  if (!response.ok) {
    throw new Error(label + " HTTP " + response.status);
  }
  try {
    return await response.json();
  } catch (error) {
    throw new Error(label + " 不是有效的 JSON");
  }
}

async function loadAll(generation) {
  const base = dataBase();
  const catalog = await readJson(await fetch(base + "/catalog.json", { cache: "no-store" }), "目錄");
  if (!catalog || !Array.isArray(catalog.seeds) || !catalog.default_selection || !catalog.phrases) {
    throw new Error("目錄缺少回放清單");
  }
  const replays = {};
  for (const seed of catalog.seeds) {
    const payload = await readJson(await fetch(base + "/replays/" + seed + ".json", { cache: "no-store" }), "seed " + seed);
    if (!payload || !Array.isArray(payload.worlds) || payload.worlds.length === 0) {
      throw new Error("seed " + seed + " 沒有回放內容");
    }
    replays[seed] = payload;
  }
  if (generation !== loadGeneration) {
    return;
  }
  state.catalog = catalog;
  state.replays = replays;
  const defaults = catalog.default_selection;
  state.memory = defaults.memory;
  state.transform = defaults.transform;
  state.q = defaults.q;
  state.seed = defaults.seed;
  state.left = defaults.left;
  state.right = defaults.right;
  state.t = 0;
  setupControls();
  if (generation !== loadGeneration) {
    return;
  }
  state.phase = "ready";
  render();
  if (generation !== loadGeneration || state.phase !== "ready") {
    return;
  }
  document.getElementById("load-status").textContent = "既有回放資料已載入。";
  document.body.dataset.phase = "ready";
  document.getElementById("retry").hidden = true;
  setControlsEnabled(true);
}

function beginLoad() {
  pause();
  loadGeneration += 1;
  const generation = loadGeneration;
  state.catalog = null;
  state.replays = {};
  clearDisplayedReplay();
  setPhase("loading", "正在載入既有回放資料。");
  loadAll(generation).catch((error) => {
    if (generation !== loadGeneration) {
      return;
    }
    failLoad("載入既有回放資料失敗。請按重新載入再試一次。");
  });
}

bindControls();
beginLoad();
