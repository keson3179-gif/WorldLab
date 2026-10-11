# worldlab

`worldlab` 是一套強調可重現性、可稽核性與有限科學聲明的生成式代理人模擬研究框架。專案從本機 LLM persona 生成與效用校準開始，之後把輸入凍結，使用純 Python 世界模擬研究歷史、社會記憶、早期擾動與長時域殘差。

目前主線已完成 **World v0.16**。舊版 README 所描述的 persona-only Phase 1 已不再代表整個專案現況。

## 目前狀態

- 最新實驗：`experiment_016_independent_seed_replication`
- 記錄規模：2,000 個正式 worlds、100 個新 seeds（100–199）、每個 world 2,000 timesteps
- 下游 LLM 呼叫：0
- 記錄狀態：complete；60/60 條舊 seed lineage 與 v0.15 一致，blend、RNG、S0、數值穩定性、safety 與六條 replay 通過
- 目前結論：在凍結的架構與 λ 網格上，H2000 onset 為 `FULL_ONSET_REPLICATION`，兩個 social architectures 的已測 band 都是 `(0.40, 0.60]`。三類分類與相鄰 λ 的正向平均差也重現。這是同一模擬架構的 stochastic replication，不是新人口、phase transition、critical point、永久路徑依賴或外部效度
- 階段性整合：[`docs/RESEARCH_SYNTHESIS_V016.md`](docs/RESEARCH_SYNTHESIS_V016.md) 索引既有證據與三張圖。它不是新的模擬實驗
- 017 已完成正式執行。見 [`docs/EXPERIMENT_017_REPORT.md`](docs/EXPERIMENT_017_REPORT.md)。設計稿仍是當時的 DESIGN_ONLY / NOT_RUN 紀錄
- 018 狀態為 COMPLETE。見 [`docs/EXPERIMENT_018_REPORT.md`](docs/EXPERIMENT_018_REPORT.md)。協議見 [`docs/EXPERIMENT_018_PROTOCOL.md`](docs/EXPERIMENT_018_PROTOCOL.md)。
- 019 狀態為 COMPLETE。見 [`docs/EXPERIMENT_019_REPORT.md`](docs/EXPERIMENT_019_REPORT.md)。協議見 [`docs/EXPERIMENT_019_PROTOCOL.md`](docs/EXPERIMENT_019_PROTOCOL.md)。
- 020 狀態為 COMPLETE。見 [`docs/EXPERIMENT_020_REPORT.md`](docs/EXPERIMENT_020_REPORT.md)。協議見 [`docs/EXPERIMENT_020_PROTOCOL.md`](docs/EXPERIMENT_020_PROTOCOL.md)。更正與封存稽核見 [`docs/EXPERIMENT_020_ERRATA.md`](docs/EXPERIMENT_020_ERRATA.md)。Experiment 021 尚未開始。
- Research Synthesis v0.20 已完成。見 [`docs/RESEARCH_SYNTHESIS_V020.md`](docs/RESEARCH_SYNTHESIS_V020.md)。它整合 Experiment 001–020 的既有證據，不是新的模擬實驗。Experiment 021 尚未開始。
- V1 Preview 可以本機回放。狀態是 `V1_PREVIEW`、`REPLAY_ONLY`、`NOT_RELEASED`。見 [`docs/V1_SCOPE_AND_ACCEPTANCE.md`](docs/V1_SCOPE_AND_ACCEPTANCE.md) 與 [`v1_demo/README.md`](v1_demo/README.md)。它引用 Experiment 020，不取代 Experiment 020，也不是正式 V1 發布。Experiment 021 維持 `NOT_STARTED`。
- 開發基線：Windows x64、CPython 3.12.10；2026-09-28 完整執行 `252 passed`

完整譜系與每一版研究問題見 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。到 v0.16 為止的證據整合見 [`docs/RESEARCH_SYNTHESIS_V016.md`](docs/RESEARCH_SYNTHESIS_V016.md)。001–020 的後續綜合見 [`docs/RESEARCH_SYNTHESIS_V020.md`](docs/RESEARCH_SYNTHESIS_V020.md)。Experiment 021 尚未開始。

## 研究管線

```text
Python-seeded traits
        │
        ▼
Persona Generator ──► Reproducibility Gate 0 ──► frozen persona_set_v1
        │
        ▼
Experiments 001–001f ──► independent place utilities ──► Utility Atlas v1
                                                            │
                                                            ▼
World v0.1 baseline ──► history/social-memory experiments ──► World v0.16
```

LLM 與 Python 的責任刻意分離：

- Persona traits 由 Python `random.Random` 與固定 seed 產生；LLM 不能修改數值特質。
- Persona profile 與早期效用校準由本機 OpenAI-compatible API 生成，並經 Pydantic schema 驗證。
- Utility Atlas 凍結後，World v0.2–v0.16 不再呼叫模型；歷史、社會記憶、選擇與介入均由可重播的 Python 規則處理。
- 每個正式實驗保留 config、manifest、來源 hash、gates、trajectory hash、replay 檢查與科學聲明邊界。

## 目錄

```text
worldlab/
├─ agents/                 # persona 生成與 audit 寫入
├─ config/settings.py      # 模型、seed 與共同執行參數
├─ experiments/            # persona／choice／utility 校準實驗
├─ llm/                    # OpenAI-compatible 本機模型 client
├─ models/                 # Pydantic schemas
├─ world/                  # 世界模擬與 World v0.1–v0.16
├─ tests/                  # 單元、lineage、replay 與 invariants
├─ data/                   # frozen inputs 與產出；大型 JSONL 不進 Git
├─ docs/EXPERIMENTS.md     # 實驗譜系與閱讀入口
├─ requirements.txt        # 直接依賴與最低版本
└─ requirements-lock.txt   # 已驗證的完整 Python 3.12 環境
```

## 安裝

需要 Python 3.11 以上；目前已驗證版本是 Python 3.12.10。在 `worldlab` 目錄執行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

若要解析允許升級的最低版本範圍，可改裝 `requirements.txt`，但必須重新跑完整測試，並刻意更新 lock file。

## 驗證

日常開發先跑約一分鐘的快速 suite：

```powershell
.\.venv\Scripts\python.exe -m compileall -q agents config experiments llm models world tests
.\.venv\Scripts\python.exe -m pytest -q -m "not slow" --durations=10
```

涉及模擬核心、lineage、frozen hashes、正式提交或 release 時，必須再跑完整 suite：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --durations=20
```

`slow` marker 只標記七個預期超過 30 秒的 deterministic cross-version lineage replay；它們不會被刪除或弱化。完整 suite 在目前基線約需 10 分鐘。測試使用 fake LLM client，不會連到 Ollama。

## Persona 再現性契約

會影響模型與 seed 的共同設定集中在 `config/settings.py`，且研究程式刻意不讀環境變數。

特質演算法版本為 `worldlab.traits.v1`：

1. `material = UTF-8("{trait_algorithm}|{master_seed}|{agent_index}")`
2. `trait_seed` 為 SHA-256(material) 前 8 bytes 的 big-endian uint64
3. `rng = random.Random(trait_seed)`
4. 依 `AgentTraits` 欄位順序各呼叫一次 `rng.random()`

欄位順序是 RNG 契約的一部分。修改 prompt 或 `BehavioralProfile` schema 時必須提高 `prompt_version`；metadata 另保留 `prompt_sha256`、`response_schema_sha256` 與 Python 版本。

第一版 persona schema 刻意不包含姓名、性別、年齡、國籍、族群、宗教或職業；模型不能自行補上人口學身份。

## 本機 LLM 任務

只有在明確需要重建 persona 或 utility artifacts 時，才啟動 Ollama 並執行 LLM entry point。例如 persona pilot：

```powershell
.\.venv\Scripts\python.exe -m experiments.persona_pilot
```

預設 endpoint 是 `http://localhost:11434/v1`，模型與 sampling 參數由 `config/settings.py` 控制。正式 run 會寫入 persona JSON、metadata 與逐次呼叫的 JSONL audit。

不要為了跑下游 World v0.2–v0.16 啟動或呼叫模型；這些版本應使用 frozen persona 與 Utility Atlas，且 manifest 的 `llm_calls` 應為 0。

## 資料與版本控制政策

`data/` 約 114 GB（111 GiB），主要由可再生的 JSONL traces 與封存 ZIP 組成。

- `data/**/*.jsonl` 與 `data/**/*.zip` 被 Git 與 Cursor 索引排除。
- 小型 config、manifest、summary、comparison、CSV 與 frozen persona JSON 保留，作為研究證據與 replay 索引。
- 已存在的 `world_v0_*`、`utility_atlas_v1` 與 calibration 輸出視為 immutable。
- 新研究必須建立新的版本化輸出目錄，不得原地覆寫上游結果。
- 先讀 config／summary／manifest，再針對性讀取大型 trace；不要遞迴載入整個資料樹。

## 協作方式

工作區根目錄的 `AGENTS.md` 與 `.cursor/rules/worldlab-collaboration.mdc` 是 Codex／Cursor 共用契約。修改程式時：

1. 先確認上游 frozen hashes 與實驗 lineage。
2. 保留其他代理人與使用者的現有變更。
3. 新增或更新聚焦測試。
4. 執行 `compileall` 與完整 `pytest`，並回報無法驗證的部分。
5. 更新 README 或實驗索引，避免研究階段再次漂移。
