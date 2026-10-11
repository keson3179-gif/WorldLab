# Research Synthesis v0.16

這份文件整理 WorldLab 到 World v0.16 為止已經凍結的證據。它是階段性索引，不是新的模擬實驗，也不建立 Experiment 017。文中的分類、門檻與數值都沿用既有輸出；本輪沒有重跑世界、沒有重建 Utility Atlas，也沒有呼叫 LLM。

關鍵數字的來源列在 [`evidence_index.csv`](research_synthesis_v016/evidence_index.csv)。圖與衍生表在 [`research_synthesis_v016/`](research_synthesis_v016/)。

## 1. 研究問題與目前定位

本專案的問題意識來自長期與虛擬觀光、Tourist Gaze，以及目的地如何在重複造訪與社會可見度中形成。這份整合不新增文獻回顧，也不把任何模擬結果寫成已驗證的真人觀光行為。

目前能夠支持的定位比較窄：在一份凍結的語意效用表上，用可重現、可控制介入的代理人模擬，檢查歷史、社會記憶與回饋混合如何留下有限時域的佔用差異。代理人有數值 traits 與場所效用，但下游世界不再請模型做選擇。因此這是一套模擬方法與內部機制證據，不是觀光者的實地效度研究。

## 2. 系統分工

責任分成兩段。

Python 以固定 seed 產生數值 traits。本機 LLM 只在早期 persona 與效用校準中填寫受 schema 約束的語意評估。Utility Atlas v1 把 persona × place-state 的效用凍結下來，SHA-256 為 `664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373`。之後的 choice、RNG、狀態轉移、memory 與 intervention 都由 Python 決定。World v0.2 以後的正式世界記錄 `llm_calls = 0`。

這幾種證據不要混在一起：

- 可重現性：同一程式、同一凍結輸入與同一 seed，軌跡 hash 可以對上。
- 測量敏感性：標籤、分數尺度或 normalization 會不會製造看起來像偏好的差。
- 機制識別：在受控對比裡，哪一種內部狀態或回饋安排足以留下殘差。
- Stochastic replication：換一組 seeds，預先固定的分類是否還在。
- 外部效度：模擬差異是否對應真人觀光行為。這一項目前沒有證據。

## 3. 實驗證據鏈

### 001–001F：直接選擇的測量問題

001c 把同一物理選項對調呈現。標成 A 與標成 B 的 pooled 選擇率差是 0.6025（C01）。001d 把標籤、位置與 schema enum 拆開：A 標籤優勢約 0.561，第一位置差約 -0.086，enum 順序差是 0（C02、C03）。直接二選一的選擇率會被任意標籤帶走。

001e 改為每個場所獨立評分，再由 Python 比較。高 crowd aversion 的低擁擠選擇率達到 0.975（C04）。001f 檢查成分是否分開移動：crowd 與 novelty 是部分可分離的 B，photography 是 C，holistic sub-scores（C05）。效用成分不能當成彼此獨立的心理量尺。

### World v0.1 與 Dynamics Calibration

v0.1 使用同步更新：所有代理人讀同一個 committed state，再一起提交移動。Dynamics Calibration 在 16 格 temperature × reconsideration 上標示操作區間。它的 summary 寫明這些標籤不是真人真實度的排序，而且這個設計沒有 human calibration（C06）。後續世界使用的 choice temperature 200 與 reconsideration rate 0.5，是技術參考點。

### 002–003：外生歷史與有限的熟悉度增強

002 比較固定的外生歷史與打亂後的歷史。熟悉度會改變軌跡，但 summary 把自己限制在描述性效果量。003 讓熟悉度只由 run 內的到訪累積。預先寫好的放大規則沒有成立：`history_reinforcement_amplified_path_dependence` 是 false（C07）。內生熟悉度沒有依該規則放大路徑差異。

### 004–006：人氣信號、記憶與校準

004 把 familiarity weight 設為 0，加入 lagged、leave-one-out 的累積人氣。結果是 `mixed pattern outside the pre-specified classes`（C08）。內生與外生 signal 的差距小於預先 margin，不能寫成內生社會回饋已經放大集體路徑依賴。

005 比較 cumulative、rolling-window 與 exponential-decay。CUM 沿用 004 的 log1p/max；rolling 與 exponential 使用 centered scores。summary 明確記錄 social weight 不能跨架構當成同等強度（C09）。因此 004 到 005 的差異同時含有記憶表示與 normalization，不能全部歸給 memory timescale。

006 在 centered architectures 與 weights 25、50、100、150 上做操作區間校準。CUM 不在這個尺度裡。選出的技術參考格是 RW50、weight 100，合格格只有 2 個（C10、C10b）。這仍是模擬操作點，不是真人校準。

### 007–009：早期介入與狀態載體

007 在 t=21 對兩個代理人做一次性 override，其後自由演化。當時的結論是早期擾動留下集體殘差（C11）。008 用兩條同樣大小、指向不同地點的替代歷史，得到有社會回饋時的持續分歧，但沒有方向保留（C12）。殘差不是「被指定的目的地鎖定」。

009 在擾動之後交換 non-social state 或 social memory。結論是殘差可由多個內部通道承載（C13）。這個 sufficiency 以後續既有動態仍在運作為前提，hybrid branches 也不是自然歷史。

### 010–011：不對稱關閉與對稱共同環境

010 從 t=22 起做兩種關閉。不對稱的 social-off 下殘差仍在；把分支社會分數換成 control field 則壓低殘差（C14）。011 讓兩條歷史進入相同的後續環境。對稱 social-off 下早期擾動衰減；共享的非零社會場會把兩條歷史同步起來。010 裡被放大的 social-off 對比，混合了早期擾動與不對稱的後處理，不能單獨識別 social-off 下的歷史持續（C15）。

### 012–013：回饋混合與 500-step replication

012 把決策用的社會分數寫成共享中點場與分支內生場的混合。λ 是模擬控制參數，不是人類自主性。在 seeds 0–49、500 steps 上，平均歷史持續的 onset 落在已測區間 `(0.20, 0.40]`（C16）。

013 用 seeds 50–99 重做這條粗略型態。兩個社會記憶架構都重現了持續，但細 band 不同：RW50 是 `(0.35, 0.40]`，EXP99 是 `(0.30, 0.35]`（C17）。這是同一架構下的 stochastic replication。

### 014–016：時域、補點與新 seeds

014 把同一批 seeds 50–99 跑到 t=2000，並要求前 500 steps 與 013 一致。兩個架構都是 `ONSET_SHIFTS_UP`（C19）。500-step 的 persistent residual 會隨視窗改變。EXP99、λ=.40 的 H2000 平均 D 是 0.01547，分類為 attenuated，但 seed 持續比例仍是 0.22（C18）。當時 H2000 的下一個已測持久點是 λ=1.00，`(0.40, 1.00)` 內部沒有測量。

015 在同一批 seeds 上補上 λ=.40、.60、.80、1.00。H2000 的已測 onset band 在 RW50 與 EXP99 都是 `(0.40, 0.60]`（C20）。相鄰 λ 的平均差為正。EXP99、H2000、λ=.40→.60 的持續集合不是完全巢狀：有 5 個 seed 退出、21 個進入（C21）。λ=.40 的平均衰減，不表示每個 seed 都消失。

016 固定架構、網格與 horizon，只把正式 seeds 換成 100–199。60 條舊 seed 軌跡先與 v0.15 對上，然後才跑 2000 個正式世界，`llm_calls = 0`（C26）。H2000 的預定 onset 判準是 `FULL_ONSET_REPLICATION`（C22）。三類分類與三個相鄰平均方向也重現。細節與沒有重現的次要格子見下一節。

## 4. 已支持的研究主張

以下主張都附適用條件。超出這些條件的說法不在本文件的支持範圍內。

1. 決策介面的任意標籤會造成測量 artifact。001c 的 A/B 呈現差是 0.6025；001d 把大部分差異留在標籤而不是 enum 順序（C01–C03）。條件是該組直接選擇介面。
2. Social signal 的表示方法會改變它在決策裡的作用。004 的累積人氣落在預先分類之外；005 記錄 centered 與 log1p/max 的 weight 不能互換（C08、C09）。
3. 在指定架構下，早期介入可以留下有限時域殘差。007 在 500 steps 看到殘差；014–016 顯示它是否跨過 0.025，取決於架構、λ 與視窗（C11、C18、C22）。
4. 共同社會場與對稱 social-off 會改變歷史殘差。011 裡，共享非零社會場同步兩條歷史，對稱 social-off 則讓早期擾動衰減（C15）。010 的不對稱比較不能單獨承擔這個識別（C14）。
5. 在已測 λ 上，平均歷史殘差呈正向劑量反應。012 的粗 onset、015 的相鄰平均差，以及 016 的三個相鄰差都為正（C16、C22）。這是相鄰已測點的方向，不是連續曲線。
6. H2000 的網格 onset 分類在新 seeds 上重現。016 的兩個架構都是 `ONSET_REPLICATED`，已測 band 仍是 `(0.40, 0.60]`，整體是 `FULL_ONSET_REPLICATION`（C22）。
7. 群體平均、個別持續比例與 seed identity 是不同結果。014 的 EXP99、λ=.40、H2000 平均已衰減，仍有 22% 的 seeds 持續（C18）。016 的主要集合轉換同時有進入與退出，而且觀察重疊與固定集合大小的隨機參考相容（C24）。

## 5. 限制與未解問題

- 世界是 20 個代理人、5 個場所、同一份 frozen Atlas。
- Technical calibration 不是真人行為校準（C06）。
- Utility component 不能當獨立心理量尺。photography 的分離等級是 C（C05）。
- 004 到 005 同時改了記憶表示與 normalization（C09）。
- 009 的 carrier sufficiency 以後續動態仍運作為前提（C13）。
- 010 的不對稱比較不能單獨識別 social-off 下的歷史持續（C14、C15）。
- 2000 steps 是有限時域。
- 網格 onset 是相鄰已測點夾出的區間，不是精確臨界點。`(0.40, 0.60)` 內部沒有測試。
- Bootstrap 與 Wilson 區間描述的是固定架構下的 seed variation，而且是 pointwise，不是多重比較下的同時 95% 覆蓋。
- matched-RNG 下，同一 seed 的 CONTROL 與 PERTURBED 是配對歷史。這個 TV 不能直接當成任意兩個獨立世界的距離。
- 尚無外部行為驗證。沒有真人觀光觀察來核對這些殘差。

全零樣本的 bootstrap 區間會退化成 [0, 0]。那只表示已觀察的 seeds 全是 0，不能證明未見 seeds 的母體平均是 0。

## 6. 015 與 016 的重現範圍

v0.15 是 seeds 50–99，n=50。v0.16 是 seeds 100–199，n=100。兩組沒有合併，也沒有用差異區間去改寫任一方的正式分類。圖 1–3 把兩組並列。

### A. 已重現

在 H2000、預定的點估計規則下，016 重現了：

- 預定 onset pattern：兩個架構都是 λ=.40 平均 D < .025，且 λ=.60、.80、1.00 的平均 D ≥ .025。
- 兩架構的已測網格區間都是 `(0.40, 0.60]`。
- 三類分類：.40 為 attenuated，.60、.80、1.00 都是 persistent residual。沒有出現 amplified divergence。
- 三個相鄰 λ 的平均差都大於 0，paired-seed 區間也都在 0 以上。
- EXP99、H2000、λ=.40→.60 的集合同時增加與退出。v0.16 是 AP=49、PA=8。

整體標籤 `FULL_ONSET_REPLICATION` 只描述這個主要 onset 判準。

### B. 邊界敏感性

RW50、λ=.60、H2000 的 v0.16 平均是 0.025925，分類為 persistent residual（C23）。它只比門檻 0.025 高 0.000925。五位小數下常寫成 0.02592，差距 0.00092。該格的 percentile 95% CI 是 [0.01857, 0.03317]，跨過分類門檻。該格中位數為 0；另有 34/100 seeds 的個別 window mean 達到 0.025。兩者是分別計算的分布摘要。

`FULL_ONSET_REPLICATION` 仍然成立，因為預定規則看的是平均點估計，不是 CI 或中位數。這個結果不是精確臨界值，也不是對邊界的高精度定位。

### C. Cohort 差異

- RW50、λ=.40、H2000：舊 cohort 的平均是 0，區間退化成 [0, 0]。新 cohort 的平均是 0.00200，3/100 個 seeds 達到個別持續線，不配對差值的區間在 0 以上。新平均仍低於 0.025，分類還是 attenuated。舊樣本全零時，bootstrap 無法估計未觀測尾端。不能只因為差值區間高於 0，就說模擬機制或母體已經改變。
- EXP99、H1000、λ=.40：舊 cohort 的平均高於 0.025，新 cohort 的平均分類是 attenuated（C25）。這是次要視窗的分類差異。
- 這兩項都不改 H2000 的主要 onset 判準，也不把兩組合併後重算分類。

### D. Identity

主要格子仍是 EXP99、H2000、λ=.40→.60。v0.16 的觀察重疊是 12，固定集合大小下的期望重疊是 12.2，中央 95% 參考區間是 [8, 16]（C24）。觀察重疊與這個隨機集合參考相容。

這不能寫成兩集合已經證明獨立，不能寫成存在特殊 seed 類型，也不能寫成已經識別集合替換的因果機制。圖 3 的兩個 cohort 使用不同 seed IDs，Jaccard 0.1875 與 0.1739 不能直接當成 identity stability 的改變。

### E. LH-B

LH-B（`DURABLE_PERSISTENCE`）只表示 H500、H1000、H1500、H2000 這四個指定視窗的 regime mean 都達到持續門檻。它不表示整段軌跡的每一步都持續，也不表示殘差永久不衰減。016 裡，兩個 social 架構的 λ=.60、.80、1.00 是 LH-B；λ=.40 是 LH-C。S0 的四個視窗平均都是 0，是 LH-A。

## 7. 圖與來源紀錄

- 圖 1：`research_synthesis_v016/figures/fig1_h2000_mean_and_persistence.svg`
- 圖 2：`research_synthesis_v016/figures/fig2_horizon_mean_d.svg`
- 圖 3：`research_synthesis_v016/figures/fig3_exp99_h2000_identity.svg`

同目錄有 PNG。左側或折線圖的平均使用 percentile bootstrap 區間；持續比例使用 Wilson 區間。圖 3 的參考區間只寫在註記裡，沒有畫成效果量的誤差線。連線只連接已測視窗。

衍生表、繪圖程式與 `source_manifest.json` 都在 `research_synthesis_v016/`。那份 manifest 記錄凍結來源的 SHA-256 與衍生檔 hash。它不是新的 experiment run manifest。重畫步驟見 [`research_synthesis_v016/REBUILD.md`](research_synthesis_v016/REBUILD.md)。繪圖套件列在 `plotting-requirements.txt`，不在核心 `requirements-lock.txt`。

## 8. 尚未執行的候選方向

下面三項只說明問題，不編號成實驗，也不在本輪執行。

1. `(0.40, 0.60)` 內部的 λ。現在的 onset 只是兩個已測點夾出的區間，而且 RW50、λ=.60 的平均距門檻只有 0.000925。更密的網格要回答的是：平均分類的跨越落在哪兩個更近的已測點之間。補密網格可提高已測點的解析度；但平均估計的抽樣不確定性仍需另行評估，不能只靠更密網格消除。
2. 把記憶表示與 normalization 拆開。004 到 005 同時改了兩件事（C09）。分開的對比要回答的是：決策差異有多少來自分數尺度，有多少來自動態記憶。它值得做，因為後面的 RW50 與 EXP99 比較都站在這個尺度選擇之上。
3. 外部行為對照的設計，而不是再加一輪相同世界。目前沒有真人選擇或目的地資料。要回答的是：這些有限時域殘差有沒有任何可觀察的對應。它值得做，因為內部 replication 不能補上這一項。
