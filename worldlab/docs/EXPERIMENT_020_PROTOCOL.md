# Experiment 020 protocol

這份干預是在看過 Experiment 018 與 019 之後制定的。它不是那兩輪之前的預註冊。

Protocol SHA-256：`de736a5a6c457aaa80378c222f6cc320485c0e27235f94cbc9cc20990f02a02d`

Seeds 1700–2699。每個 seed 有 18 個 matched conditions、四條來源時程，以及一個 S0 donor。
Weights 從 018 的 weight_hex 讀取。q05、q15、q25 是校準標籤，不保證 recipient 的實際 Q 等於標籤。
E 使用 recipient 自己截至 t−1 的 shadow memory。O 使用同 seed、weight 0 的 S0 donor 截至 t−1 的 leave-one-out score，再乘上該 condition 的固定 weight。
切換點是 decision t=51，發生在該步所有 agents 決策之前。切換不清空 memory、不複製 donor memory、不重置位置、不消耗額外 RNG。
四條路徑是 EE、EO、OE、OO。branch ID 不進入 choice seed。
主要結果是 t=101–500 累積 visit share 的 HHI。對照是 EE−OE、EO−OO、EE−EO、OE−OO，以及 interaction EE−EO−OE+OO。
早期形成是 18/20 的 20 步區間在 committed t=50 以前確認。維持資格只看 commits t=31–50，並且在切換前固定。
Bootstrap 使用分析種子 20261002 的本地 Random，10000 次，索引 0 是 choice seed 1700，percentile indexes 249 與 9749。
歷史 lineage 57 條必須與 018 的 trajectory hash exact match。額外 replay 的浮點容差是 1e-12。
這不是等 Q 的純 feedback 效果，也不是機制貢獻百分比。外生 social field 不表示 occupancy 與 base utility 沒有 feedback。
