# Experiment 017 設計稿：Memory Representation × Score Transformation

狀態：**DESIGN_ONLY / NOT_RUN**。

本文件只固定研究問題、歷史公式、六格草案、兩層比較與未來執行條件。本輪沒有建立 `world/experiment_017.py`，沒有建立 `data/worlds/world_v0_17/`，沒有執行世界模擬，也沒有呼叫 LLM。

草案協議見 [`experiment_017_design/protocol_draft.json`](experiment_017_design/protocol_draft.json)。它是設計草案，不是已執行的 config 或 run manifest。

## 研究問題

Experiment 004 到 Experiment 005 同時更換了記憶表示與 social score 的轉換。既有差異不能全部歸因於記憶時間尺度。

本設計要回答三件事：

- A. 固定同一種記憶表示時，更換 score transformation 如何改變選擇機率與閉環動態。
- B. 固定同一套 score transformation 時，更換記憶表示如何改變上述結果。
- C. transformation 的效果是否依賴記憶表示，也就是 memory × transformation interaction。

六格只涵蓋 CUM、RW50 與 EXP99（decay = 0.99）。它不涵蓋 RW20、EXP95，也不代表整個 Experiment 005。

## 不能回答的問題

LOGMAX 與 ZRAW 的差異同時包含非線性轉換、縮放與中心化。這組對照沒有分離出純尺度效果。

記憶對照同時改變時間加權與累積量。memory contrast 不是純 timescale effect。

本設計不解釋 Experiment 016 的 H2000 onset。本輪不加入 λ、早期介入或 paired-history persistence。

Layer A 與 Layer B 的數字不能相減，也不能被寫成 feedback 的因果占比或中介效果。interaction 是指定規則下的配對對比，不是可分配的貢獻百分比。

未觀察到差異只表示該對照沒有顯示差異，不構成等價證明。weight = 150 只為了與 004/005 對齊，不表示兩種 transformation 的有效影響強度相等。seeds 0–49 是既有 cohort，不是 held-out replication。

## 歷史公式與程式來源

共同輸入是 `r[i,p,t]`：agent `i` 在場所 `p`、決策時刻 `t` 看到的 leave-one-out 記憶，只含截至 `t−1` 的已提交 presence。

| 項目 | 程式 |
| --- | --- |
| 場所順序 | `world/geography.py` 的 `PLACE_IDS`：`site_01` … `site_05` |
| LOGMAX | `world/experiment_004.py` `popularity_scores` |
| CUM 記憶 | 同檔 `empty_social`、`commit_social`；`experiment_005.py` 的 CUM 呼叫 `simulate_world` |
| RW50 | `world/experiment_005.py` `RollingMemory`，`window = 50` |
| EXP99 | 同檔 `DecayMemory`，`decay = 0.99` |
| ZRAW | 同檔 `centered_scores`、`clip_score`、`_population_sd` |
| softmax | `world/experiment_004.py` `_select`；005 直接沿用 |
| social utility | 004 `_utilities(social_weight)`；005 `_utilities` 使用模組全域 `SOCIAL_WEIGHT` |
| 閉環 late HHI | 005 `_world_metrics`：`_presence(series, 101, timesteps)` 後 `_share`、`_hhi` |

LOGMAX：

- `a[p] = log1p(r[p])`
- `max(a) = 0` 時五個 score 都是 0
- 否則 `score[p] = a[p] / max(a)`

ZRAW：

- 五個 `r` 全相等時，五個 score 都是 0
- 否則 `z[p] = (r[p] − mean(r)) / population_sd(r)`，分母除以 5
- `score[p] = clip(z[p], −2, 2)`

實作禁止在 ZRAW 前加 `log1p`、在 LOGMAX 前改用 visit share、加入 pseudocount 或 epsilon、改成 sample SD、對 EXP99 再乘 `1−decay`，或為了新設計改寫舊函式。

### 時序

初始 CUM、RW 與 EXP 記憶都是零。`simulate_world` 與 `simulate_memory` 都先讓全體 agent 決策，再 `commit`。logged timestep 1 使用的記憶是零。初始位置只進入當下 occupancy，不計入 social memory。

RW commit 先加入本步 presence，若長度超過 window 再移除最舊的一筆。決策當下看到的是此前最多 50 筆 commit。

EXP commit 是 `decay * previous + new presence`。新 presence 在進入的那一步不被 decay，也沒有 `(1−decay)` 係數。own memory 用同一順序更新，loo 再減去自己的量。

004 `popularity_scores` 在 leave-one-out 小於 0 時拋出 `RuntimeError`，因此不會對負值做 `log1p`。005 `centered_scores` 沒有這個檢查。由合法 commit 產生的記憶量維持非負。兩支程式都沒有先檢查非有限值。未來執行若在呼叫舊函式之後看到非有限 score，應停止該次執行，且不改舊函式。

Agent 以排序後的 id 更新。softmax 依 `PLACE_IDS` 累加機率。每次 reconsider 只抽一次 `random.Random(choice_seed(...)).random()`，同一 draw 同時決定 social destination 與 base destination。

### 已核對的歷史差異

1. 005 的 CUM 沿用 004 LOGMAX；RW 與 EXP 改用 ZRAW。config 已標 `weights_not_equivalent_across_architectures = true`。同一數字 150 不是等強處理。
2. 005 `_utilities` 沒有 weight 參數，一律乘全域 `SOCIAL_WEIGHT`（150）。004 `_utilities` 在 weight 為 0 時把 social term 設為 0。S0 必須走 004 這條路徑。
3. 004 的 late window 是累積 social snapshot 的 `snapshots[500] − snapshots[100]`，對 CUM 對應 logged commit 101–500。005 的 `late_counts` 是 `series[101]` 到 `series[500]` 的 occupancy 總和，同樣是 t = 101–500，共 400 個位置快照。Layer B 採用 005 的 series 定義。RW/EXP 的 memory mass 會過期或衰減，不能拿來代替 visit presence。
4. 這是累積 share 的 HHI，不是逐步 occupancy HHI 的平均，也不是 t = 401–500。
5. 005 `_absorb` 的 centered 條件 `abs(score) >= 1.9`，以及 LOGMAX 的 `score > 0.95`，是 near-boundary 診斷。clip 門檻是 ±2。`abs(score) >= 1.9` 的次數不是 clipping 實際改寫 score 的次數。
6. `world_states.jsonl` 的 timestep `t` 是該步 commit 之後的位置。Layer A 在決策 `t` 使用 timestep `t−1` 的 origin 與 occupancy，記憶只吸收 timestep 1 到 `t−1` 的位置。
7. `social_signal_reference_s0.jsonl` 存的是 S0 那次執行的 CUM × LOGMAX score，不是六格共用的 score。Layer A 要從 S0 位置重建六種 score。
8. ranking change 以全部 decision 為分母。destination change 只計 reconsidered，分母是 reconsidered 次數。004 另記錄平均絕對 social utility；softmax 對共同位移不變，該平均值不代表場所區辨強度。
9. 005 讀取 004 S150 時，排序後取前 501 列再檢查長度。未來的 S0 讀取要更嚴格：timestep 必須恰好是 0 到 500 各一列，否則停止。本輪沒有掃描該 JSONL 內容。

004 manifest 的分類標籤維持原記錄 `mixed pattern outside the pre-specified classes`。本設計不改分類、不重跑 001–016，也不沿用 005 的 architecture classification 作為主要結論。

## 六格

| 格子 | 記憶 | transformation | 位置 |
| --- | --- | --- | --- |
| CUM × LOGMAX | 累積 | LOGMAX | 已存在：005 CUM，並與 004 S150 相同 |
| CUM × ZRAW | 累積 | ZRAW | 未來新增 |
| RW50 × LOGMAX | window 50 | LOGMAX | 未來新增 |
| RW50 × ZRAW | window 50 | ZRAW | 已存在：005 RW50 |
| EXP99 × LOGMAX | decay 0.99 | LOGMAX | 未來新增 |
| EXP99 × ZRAW | decay 0.99 | ZRAW | 已存在：005 EXP99 |

另設 S0 reference：`social_weight = 0`，對應 004 S0。weight 為 0 時 social term 為 0，軌跡不應隨記憶或 transformation 改變。這是動態等價閘門，不是第六種 social 結果。

橋接設定固定為：20 agents、5 places、500 timesteps、`choice_temperature = 200`、`reconsideration_rate = 0.50`、`familiarity_weight = 0`、六個 social cells 的 `social_weight = 150`、S0 為 0、seeds 0–49、synchronous two-phase、既有 stateless RNG 與場所順序、Utility Atlas v1、`llm_calls = 0`。

Atlas SHA-256：

`664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373`

本輪不增加 weight grid，也不為了放大差異而改 weight。

六格能分開的是：在這三種記憶與兩種既有轉換之下，轉換效果、記憶效果，以及兩者的交互對照。它仍不能分開非線性、縮放與中心化，也不能把記憶效果收成單一時間尺度。

## 兩層比較

### Layer A：共同歷史上的離線決策

使用 Experiment 004 已保存的 S0 軌跡。同一 seed、agent、timestep 使用相同 origin、occupancy 與 Atlas base utility。三種記憶只吸收這條已提交的 S0 歷史。每種記憶分別套 LOGMAX 與 ZRAW。假想選擇不回寫 S0，也不寫入 shadow memory。

資料位置是 `data/worlds/world_v0_4/world_states.jsonl`（本輪只確認檔案存在，約 66.7 MB）。未來以串流過濾 `condition_id = S0`，並先用 manifest 內已記錄的檔案 SHA-256 核對。不讀 `agent_actions.jsonl`（約 2.3 GB）。本輪沒有計算這批 Q。

主要指標：

`Q = 0.5 × Σp |P_social(p) − P_base(p)|`

`P` 是重新考慮時的五選項 softmax。每個 agent × timestep 都計算這個條件機率。未 reconsider 的留在原地 one-hot 不進入 Q。先在每個 seed 內平均，再跨 seeds 彙整。

Q 是選擇機率差異，不是配對世界的 occupancy TV。既有 0.025 persistence 門檻不適用。

次要診斷可以包括 score 的跨場所 SD 與 range、`SD(social utility) / choice_temperature`、記憶總量與零變異比例、ranking change。若報告 matched-draw destination change，分母只含 reconsidered，而且 social 與 base 必須共用同一個 draw。

### Layer B：各自生成歷史的閉環

六格各自產生決策與 memory。比較的是整套規則的閉環結果。

主要結果沿用 005 的 late visit-share HHI，窗口是 t = 101–500 的累積 presence。

次要結果可沿用 late visit-share entropy、late dominant share、movement、alternation、consecutive occupancy TV，以及原有 safety 診斷。005 的 architecture classification 不作為本設計的主要結論，也不用來重分類舊結果。

## Contrasts、分析單位與 bootstrap

兩層分開分析。對每個 seed 的主要結果 `Y`：

1. 同一 memory：`ΔT(m) = Y(m, ZRAW) − Y(m, LOGMAX)`
2. 同一 transformation：`RW50 − CUM`、`EXP99 − CUM`
3. interaction：`I(m) = ΔT(m) − ΔT(CUM)`，`m` 為 RW50 或 EXP99

bootstrap 以 seed 為重抽單位。索引是 `list[list[int]]`：外層 10,000 列，每列 50 個索引，範圍是 `0 <= index < 50`。每一列都是有放回抽樣，同一個 seed 可以在同一列出現多次。產生器是局部的 `random.Random(20260928)`，不碰全域 RNG，也不碰 choice RNG。

每個 replicate 取出該列的 50 個索引。同一層裡的所有 cells 共用這一列，先算各 cell 的重抽平均，再從配對平均計算 transformation、memory 與 interaction。Layer A 與 Layer B 仍分開，不混合結果。agent × timestep 不是獨立樣本。

百分位索引維持 `int(0.025 × 9999) = 249` 與 `int(0.975 × 9999) = 9749`。區間註明 pointwise。不以多個區間是否跨過 0，事後產生整體成功標籤。

本輪只對合成 seed 序列做了抽樣檢查。尚未對正式 Q 或 HHI 執行 bootstrap。

## 三種驗證

trajectory hash 必須完全一致。score SD 等浮點診斷若要比較，須另寫明容差；hash 閘門不用容差放行。失敗就停止並記錄失敗條件，不放寬 hash，也不修改歷史來源。

### 歷史 lineage：200 條

四個歷史條件是 CUM_LOGMAX、RW50_ZRAW、EXP99_ZRAW 與 S0。每個條件 seeds 0–49，共 4 × 50 = 200 條。

未來要用新執行器重現完整 t = 0–500 軌跡，並與對應歷史 canonical trajectory hash exact match。本輪對兩份舊 manifest 的 hash 對照只說明既有紀錄彼此一致，不能代替這 200 條新執行器驗證。

200 條全部通過之前，不模擬 150 個新閉環條件，也不開始實質 Layer A/B 分析。

S0 動態等價仍要保留。這 50 條 S0 已算在 200 裡，必須等於 004 S0。`social_weight = 0` 時 social term 為 0，因此不為六個 score cells 再加一批 social-off worlds。額外執行數是 0。

### 指定歷史子集：8 條

四個歷史條件 × seeds 0 與 49，共 8 條。這是上面 200 條的子集，`additional_count = 0`。它不是 8 個新 worlds，只驗證這 8 條也不夠。

### 新格子產出後 replay：6 條

主執行完成後，從原始初始狀態重跑：

- CUM_ZRAW，seeds 0 與 49
- RW50_LOGMAX，seeds 0 與 49
- EXP99_LOGMAX，seeds 0 與 49

比對目標是本次主執行已保存、並將寫入本輪 manifest 的 canonical trajectory hash。不使用另一格的歷史 hash，不重用主執行的結果物件，不用 replay hash 覆寫主執行 hash，也不把同一份檔案重讀兩次當成 replay。六條全部 exact match，正式執行才可標成驗證完成。本輪只固定清單，不執行 replay。

### 執行順序

來源完整性核對 → 固定正式 config → 完整歷史 lineage 與既有等價閘門 → 全部通過 → Layer A/B 正式工作 → 指定的新格子 replay → 最終驗證與 manifest 完成。

下表的 seed 0 與 seed 49 是歷史 canonical hash，供未來 200 條 lineage 使用。本輪只讀了 manifest 裡的這些紀錄。

| Anchor | 來源 | seed 0 trajectory SHA-256 | seed 49 |
| --- | --- | --- | --- |
| CUM × LOGMAX | 005 `CUM:{seed}`，004 `S150:{seed}` | `7c7e4d9c40836a202f06ff9bdf2e15b7e5c1957537cab5b1886bb4a77d1c1e96` | `c15f6df1599a29b827c1cadaee7f50f27fc1bcf1e4a6a801ad140ab99eb75259` |
| RW50 × ZRAW | 005 `RW50:{seed}` | `2307fe77c9e80f8f3b66ddd4d7e20a46b54146ad382e4e76660e4e6a2da36b25` | `845e7ccda3533c14f9affecb97035a7ec414e3686662e7e048f8ce195fca2cfd` |
| EXP99 × ZRAW | 005 `EXP99:{seed}` | `6fe99a3fe69ac495f7fbb65fc5b89bdf2657dab984205607071890db1cc3f2fe` | `ddfbad38a89a2940685807de0f301c8a1b095049030a688c6df73fc6f3cf9eac` |
| S0 | 004 `S0:{seed}` | `610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955` | `d3943a126e34a9c5adfdeab25fb5360b57b91ebe661148c08adc8cbfc1436297` |

004 config SHA-256 是 `341f9ed79068e87930e4c4f9553bcef4a512ecd8716c14d92cfd5a2335e353f1`。005 config SHA-256 是 `0ec8c27c9ae974fb1acf329c2dc8ba6022787ae1864693cd2c71937fed0c1f32`。

## 計數與輸出規劃

邏輯條件、首次新增條件、lineage 執行與 replay 分開計數：

| 項目 | 數量 |
| --- | --- |
| 邏輯 world 條件 | 350 |
| 歷史已存在條件 | 200 |
| 新閉環條件 | 150 |
| 未來完整歷史 lineage 執行 | 200 |
| 指定歷史子集 | 8，已包含在 200 內 |
| 新格子產出後 replay | 6，屬額外驗證執行 |
| Layer A | 50 條既有 S0 歷史的離線計算 |
| 本輪實際執行的正式 worlds | 0 |

這 350 個條件不是 350 個新 worlds。8 條子集不再另計。6 條 replay 也不算進 150 個新條件。

未來輸出目錄若建立，必須拒絕覆寫非空目錄，寫 manifest，以 1 MiB 串流計算 SHA-256，並在正式執行前固定 config。來源目錄與新輸出分開。不保存可重建的大型 action JSONL。

磁碟與時間是假設，不是本輪實測：

- 004 的 `agent_actions.jsonl` 約 2.31 GB / 200 worlds，約 11.5 MB/world。若 150 個新世界照這方式存 action，約 1.7 GB。未來設計不採用這項輸出。
- 004 `world_states.jsonl` 約 66.7 MB / 200 worlds。若只存 compact metrics、hash 與 seed 摘要，並以每筆邏輯條件低於 50 KB 估計，350 筆約低於 20 MB。
- 016 的 2,000-step compact run 約 5,960 秒 / 2,000 worlds。若時間大致隨步數比例下降，500 steps 約 0.75 秒/world。200 次 lineage、150 次新格子與 6 次 replay 約數分鐘量級。這個外推尚未用 017 程式測量。

## 已驗證的數學性質

檢查程式是 `docs/experiment_017_design/formula_checks.py`。它呼叫既有的 `popularity_scores`、`centered_scores`、`RollingMemory`、`DecayMemory`、`commit_social` 與 `_select`。結果寫在 `formula_report.json`。

1. Softmax 對共同位移不變。同一決策的五個 utility 加上同一常數，`_select` 的機率不變。LOGMAX 分數為正、ZRAW 分數可負，不能因此解釋成吸引與排斥兩種心理機制。social utility 的平均絕對值也不是區辨強度。
2. 全零輸入時，LOGMAX 與 ZRAW 都是全零。相等且為正的 `r`：LOGMAX 是全 1，ZRAW 是全 0。在同一 base utility 上，兩者的 softmax 機率相同，此時 Q 為 0。
3. 非退化時，`c × r`（`c > 0`）在精確算術下不改變 ZRAW。合成輸入 `1…5` 與 `3 × (1…5)` 的 ZRAW 差只到約 `2×10⁻¹⁶`，屬於浮點殘差。LOGMAX 的改變是實質的。場所比例固定時，LOGMAX 仍受累積量影響。這是公式性質，不是實驗證據。ZRAW 使用 population SD，與 sample SD 不同。
4. 五個等權數值的非退化 population z 滿足 `|z| ≤ sqrt(4) = 2`。一端為尖峰、其餘相等時，尖峰的 `|z|` 等於 2。因此精確算術下 clip ±2 不改寫 z。0 到 8 的整數格點上，clip 與未 clip 的最大差約 `4×10⁻¹⁶`，同時仍有 `|z| ≥ 1.9` 的格子。該 near-boundary 計數不是 clip 改寫次數。非退化 z 的 population SD 為 1，所以 score SD 接近 1 不是獨立的行為發現。數值殘差沿用舊的 `clip_score`，不改演算法。
5. 小型合成歷史確認：commit 前記憶為零；CUM 的 leave-one-out 在 commit 之後才納入他人 presence；RW window 先加入再移除最舊筆；EXP99 第一個雙人 commit 使 total 成為 2，而不是 `1−0.99` 的縮放。負的 leave-one-out 會被 `popularity_scores` 拒絕。
6. 合成 bootstrap 索引是 10,000 × 50。`A = [0, 1, 4]`、索引 `[2, 2, 1]` 的平均是 3。每個 seed 的 `B = A + 3.5` 時，10,000 列配對平均差都回到 3.5。這不是正式 Q 或 HHI 的 bootstrap。

## 尚未解決的實作問題

這些問題不阻止把設計寫成未來的獨立模組，但正式實作時必須照下面處理：

- 新模組要組合既有 memory 物件與 `popularity_scores` / `centered_scores`，不修改 004 或 005。EXP 的 raw 是浮點數；`popularity_scores` 可直接吃合成的非負浮點 leave-one-out，整數歷史必須與 CUM 路徑一致。
- S0 與 CUM 的 weight 走 004 `_utilities`。005 `_utilities` 不能用來做 weight 0。
- 若未來程式暫時改到 `SOCIAL_WEIGHT` 全域值，必須在成功與例外路徑都還原。
- 浮點誤差若使 `|z|` 略大於 2，仍使用舊的 `clip_score`，不改演算法。lineage 的 trajectory hash 維持 exact match。
- Layer A 的 S0 讀取要串流、核對 manifest 檔案 hash，並在 timestep 集合不完整時停止。
- 非有限 score 由新執行器停止，不改舊函式。

## 本輪沒有做的事

沒有正式 world，沒有 LLM/Ollama 呼叫，沒有改 `data/`，沒有重跑 Experiment 001–016，沒有修改既有分類、門檻或 Research Synthesis v0.16 產物，也沒有做完整歷史資料的實質分析。完整 slow suite 本輪不要求重跑，因為模擬行為沒有改變。
