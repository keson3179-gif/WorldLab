# Worldlab experiment index

這份索引記錄已存在的研究譜系、每個階段的角色與建議閱讀順序。正式結果以各資料目錄中的 config、summary 與 run manifest 為準，程式 docstring 與測試則定義可執行契約。

## 建議閱讀順序

針對任一 World 版本，依序閱讀：

1. `data/worlds/world_v0_N/world_v0_N_config.json`
2. `data/worlds/world_v0_N/experiment_0NN_summary.json`
3. `data/worlds/world_v0_N/run_manifest.json`
4. `world/experiment_0NN.py`
5. `tests/test_experiment_0NN.py`

大型 `*.jsonl` 是逐步 audit／state trace，只有在 summary 與 manifest 不足以回答問題時才針對性開啟。

## Persona 與效用基礎

- **Persona pilot**：Python 以固定 seed 產生 traits，本機 LLM 只填寫 schema-constrained behavioral profile。
- **Reproducibility Gate 0**：獨立重跑 persona generation，比對 traits、profiles、schema violations 與檔案 hashes。
- **Frozen persona set v1**：從 Gate 0 產物選出固定 persona，供後續決策與世界模擬使用。
- **Experiment 001**：測試 `crowd_aversion` 是否改變低擁擠地點選擇。
- **Experiment 001b**：在固定 trade-off cost ladder 上校準 crowd-avoidance response。
- **Experiment 001c**：鏡像顯示同一物理選項，測量 place-choice order invariance。
- **Experiment 001d**：拆開位置、A/B label 與 schema enum order 的影響。
- **Experiment 001e**：每個地點獨立、無 label 地估計 utility，再由 Python 比較。
- **Experiment 001f**：固定其他欄位，檢查 trait–attribute separability。
- **Utility Atlas v1**：凍結可內容定址的 persona × place-state utility scores，成為下游共同輸入。
- **Dynamics Calibration 001**：固定下游 choice temperature、reconsideration rate 與初始條件。

## World lineage

### World v0.1 — synchronous baseline

所有 agents 從同一 committed state 評估候選地點，再同步提交移動。效用呼叫以 prompt hash cache，建立最初世界軌跡與分析。

### World v0.2 — fixed-reference history dependence

使用 frozen Utility Atlas，在 Python 端加入 familiarity；比較固定 favorite history、shuffled history 與 reference regime。

### World v0.3 — endogenous familiarity reinforcement

所有 agents 從零歷史開始，visit counts 只由 run 內 committed moves 累積，測試內生 familiarity 是否放大早期差異。

### World v0.4 — endogenous social popularity

固定 familiarity weight 為 0，以 lagged、leave-one-agent-out popularity 建立社會回饋，並包含 exogenous signal control。

### World v0.5 — social-memory architectures

比較 cumulative、rolling-window 與 exponential-decay social memories；centered architectures 使用共同尺度。

### World v0.6 — social weight × memory timescale

在多個 social weights 與四種 centered memory architectures 上做 factorial calibration，建立穩定性與 reference thresholds。

### World v0.7 — controlled early perturbation

在 t=21 對固定 agents 做一次性 override，之後自由演化；比較 S0、RW50 與 EXP99 是否保留早期差異。

### World v0.8 — symmetric alternative histories

建立兩個同樣大小但指向不同地點的 early overrides，直接比較替代歷史 forks。

### World v0.9 — state-carrier decomposition

在 t=21 後交換 non-social state 或 social memory，分解長期差異由哪一類狀態承載。

### World v0.10 — feedback-closure necessity

從 t=22 起切斷 branch-specific social feedback 或關閉全部 social influence，測試 closed-loop feedback 是否必要。

### World v0.11 — symmetric post-perturbation environments

讓 control 與 perturbed history 進入相同的 endogenous、shared-donor 或 social-off 環境，移除環境不對稱解釋。

### World v0.12 — feedback-autonomy dose response

決策 social score 混合 shared midpoint field 與 recipient branch 的 endogenous field；λ 是模擬控制參數，不代表人類 autonomy。

### World v0.13 — held-out onset replication

使用 seeds 50–99，在 `(0.20, 0.40]` 內更細的 λ grid 重做 onset 分析；thresholds、介入與 grid 不因結果調整。

### World v0.14 — long-horizon persistence

從 t=0 重跑至 t=2000，且前 500 steps 必須與 v0.13 完全一致。主要視窗為 H500、H1000、H1500、H2000。

記錄結果：

- 1,000 worlds、50 held-out seeds、2,000 timesteps、0 LLM calls
- 所有 lineage、replication、blend identity 與 numerical-stability gates 通過
- RW50 的 horizon onset 是已測 λ 上的第一個達標點，不是插值臨界值。H500 的 band 是 `(0.35, 0.40]`；H1000–H2000 的 band 是 `(0.40, 1.00]`。後三個 horizon 的 onset λ＝1.00 只表示下一個已測點才達標；`(0.40, 1.00)` 內部尚未測試。
- EXP99 的 H500 band 是 `(0.30, 0.35]`，H1000 是 `(0.35, 0.40]`，H1500–H2000 是 `(0.40, 1.00]`。onset λ＝1.00 同樣不是精確門檻。
- 兩個 social-memory architectures 均為 `ONSET_SHIFTS_UP`

因此，500-step persistent residual 對 horizon 敏感。正式分類由 regime mean 決定，區間只補充不確定性。群體平均低於 0.025 並不表示每個 stochastic history 都已消失。EXP99 在 λ＝0.40、H2000 的平均 TV 是 0.01547，但 11/50 seeds 的個別 window mean 仍 ≥ 0.025。EXP99 在 λ＝1.00、H2000 則是 49/50 seeds 持續，群體平均由幾乎整個 tested cohort 支撐。這仍只是 2,000 timesteps 內的有限時域結果。

### World v0.15 — long-horizon dose fill-in

在固定 2,000 steps 上補測 λ＝0.40、0.60、0.80、1.00。Seeds 50–99 仍是相對 Experiment 012 的 held-out cohort，不是本輪新取得的獨立驗證。正式分類只看 regime mean；seed rate 與區間分開報告。600 條 λ＝0.40／1.00 與 S0 anchors 的 t＝0–2000 trajectory hash 先與 v0.14 完全一致，之後才模擬 λ＝0.60、0.80。

記錄結果：

- 1,000 worlds、0 LLM calls；anchor、blend、RNG、S0 equivalence、數值穩定性、safety 與 selected replay 都通過
- H2000 的網格 onset 在兩個 social architectures 都是 `(0.40, 0.60]`。這是相鄰已測點夾出的區間，`(0.40, 0.60)` 內部沒有測試，也不是 critical λ
- RW50：H500 的最小已測 λ＝0.40 已經達標，下界未知；H1000–H2000 的 band 是 `(0.40, 0.60]`
- EXP99：H500 與 H1000 的最小已測 λ＝0.40 已經達標，下界未知；H1500–H2000 的 band 是 `(0.40, 0.60]`
- H2000 mean TV：RW50 為 0、0.03523、0.06920、0.09269；EXP99 為 0.01547、0.03511、0.06739、0.08323。λ＝0.40 的平均低於 0.025，λ＝0.60 起為 persistent residual。相鄰 λ 的 paired-seed 平均差，95% 區間都落在 0 以上
- EXP99、λ＝0.40、H2000 的平均雖衰減，仍有 11/50 seeds 持續。λ＝1.00 是 RW50 48/50、EXP99 49/50，群體平均由幾乎整個 tested cohort 支撐
- 從 λ＝0.40 到 0.60，H2000 的持續集合在 EXP99 是替換：11 個持續 seeds 裡有 6 個留在 λ＝0.60，另有 21 個新出現。Seed ID 只描述集合是否保留、增加或替換

這仍是 2,000 timesteps 的有限時域結果，不是永久路徑依賴、phase transition、critical point、已證明的 metastability、human realism 或外部效度。

### World v0.16 — independent-seed replication

架構、λ 網格與 2,000-step horizon 固定為 v0.15；正式樣本只換成 seeds 100–199。判準在正式結果出現前寫入 config。Seeds 50–99 只用於 60 條程式 lineage 與事後的不配對 cohort 比較，不併入本輪分類。

記錄結果：

- 2,000 個正式 worlds、100 個新 seeds、0 LLM calls。Seeds 50、75、99 的 60/60 條 t＝0–2000 trajectory hash 與 v0.15 一致。新 seeds 沒有舊軌跡可對，只通過演算法契約、內部 gates 與自身 replay
- H2000 主要 onset：RW50 與 EXP99 都是 `ONSET_REPLICATED`，實際網格 band 仍是 `(0.40, 0.60]`。整體標籤是 `FULL_ONSET_REPLICATION`。這個標籤只描述主要 onset 判準
- 三類分類也重現：λ＝0.40 為 attenuated，λ＝0.60、0.80、1.00 都是 persistent residual。沒有出現 amplified divergence
- 相鄰 λ 的三個平均差都為正，paired-seed 95% 區間也都在 0 以上。標籤是 `ALL_ADJACENT_DIRECTIONS_REPLICATED`
- H2000 mean TV：RW50 為 0.00200、0.02592、0.06154、0.09291；EXP99 為 0.01491、0.04167、0.06106、0.08216。持續 seeds 為 RW50 3/100、34/100、75/100、96/100，EXP99 20/100、61/100、82/100、97/100
- 主要 identity 診斷是 EXP99、H2000、λ＝0.40→0.60：AP＝49、PA＝8，持續集合同時增加與退出。固定集合大小下的期望交集是 12.2，觀察值 12，hypergeometric 中央 95% 參考區間是 [8, 16]。這不是效應的信賴區間
- 舊 cohort 對新 cohort 的 H2000 差值，多數格子的不配對區間包含 0。RW50、λ＝0.40 的新平均高出舊 cohort 的精確 0，該格差值區間在 0 以上，但新平均 0.00200 仍是 attenuated。區間包含 0 本身不是 replication 成功
- EXP99 在 H1000、λ＝0.40 的新 cohort 平均已低於 0.025，所以該 horizon 的 onset 狀態與舊 cohort 的「最小已測 λ 已達標」不同。這不改變 H2000 的主要判準

這仍是固定架構下的 2,000-step stochastic replication，不是新人口、獨立模型、永久路徑依賴、critical point、phase transition、已證明的 metastability、真人行為效度或外部效度。

階段性索引見 [`RESEARCH_SYNTHESIS_V016.md`](RESEARCH_SYNTHESIS_V016.md)。該文件只整理既有證據、重現範圍與限制，不改變 v0.1–v0.16 的分類或凍結結果。

### Experiment 017 — design review

017 已完成正式執行。設計稿見 [`EXPERIMENT_017_DESIGN.md`](EXPERIMENT_017_DESIGN.md)，仍保持撰寫當時的 DESIGN_ONLY / NOT_RUN。結果見 [`EXPERIMENT_017_REPORT.md`](EXPERIMENT_017_REPORT.md)。

### Experiment 018 — mean decision-impact matching

018 狀態為 COMPLETE。協議見 [`EXPERIMENT_018_PROTOCOL.md`](EXPERIMENT_018_PROTOCOL.md)。結果見 [`EXPERIMENT_018_REPORT.md`](EXPERIMENT_018_REPORT.md)。Experiment 001–017 的輸出沒有因這次執行而改寫。

### Experiment 019 — closed-loop decision impact

019 狀態為 COMPLETE。這是看過 018 之後的回溯稽核。協議見 [`EXPERIMENT_019_PROTOCOL.md`](EXPERIMENT_019_PROTOCOL.md)。結果見 [`EXPERIMENT_019_REPORT.md`](EXPERIMENT_019_REPORT.md)。

### Experiment 020 — social-field source switching

020 狀態為 COMPLETE。這是看過 018 與 019 之後的來源切換干預。協議見 [`EXPERIMENT_020_PROTOCOL.md`](EXPERIMENT_020_PROTOCOL.md)。結果見 [`EXPERIMENT_020_REPORT.md`](EXPERIMENT_020_REPORT.md)。報告更正與封存稽核見 [`EXPERIMENT_020_ERRATA.md`](EXPERIMENT_020_ERRATA.md)。Experiment 021 尚未開始。

### Research Synthesis v0.20

v0.20 綜合已完成。見 [`RESEARCH_SYNTHESIS_V020.md`](RESEARCH_SYNTHESIS_V020.md)。它整理 Experiment 001–020 的既有證據、證據索引與四張圖，不新增世界、不重新 fitting、不重算正式 bootstrap。Experiment 021 尚未開始。v0.16 綜合與其衍生輸出保持原樣。017 的設計稿仍是當時的 DESIGN_ONLY / NOT_RUN 紀錄。

### WorldLab V1 Preview

V1 Preview 是既有軌跡的本機回放，狀態為 `V1_PREVIEW`、`REPLAY_ONLY`、`NOT_RELEASED`。範圍與本輪驗收見 [`V1_SCOPE_AND_ACCEPTANCE.md`](V1_SCOPE_AND_ACCEPTANCE.md)，操作紀錄見 [`V1_PREVIEW_REPORT.md`](V1_PREVIEW_REPORT.md)。啟動方式見 [`../v1_demo/README.md`](../v1_demo/README.md)。它引用 Experiment 020 的三個 gate seeds，不取代 Experiment 020，也不是正式 V1 發布。Experiment 021 維持 `NOT_STARTED`。

## 輸出契約

正式 World experiment 應至少保存：

- 完整 config 與 `config_sha256`
- frozen input／上游 lineage hashes
- primary summary 與預先定義的 thresholds
- `run_manifest.json` 與 `complete` 狀態
- trajectory hashes 與 selected exact replays
- gate 結果、數值穩定性檢查與 `llm_calls`
- 清楚的 scope sentence，區分模擬控制參數與人類／集體層級主張

既有版本化輸出不可原地修改。若研究問題、參數 grid、threshold、horizon 或機制改變，應建立下一個 World 版本。
