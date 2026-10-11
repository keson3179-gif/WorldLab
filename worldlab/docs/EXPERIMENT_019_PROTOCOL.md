# Experiment 019 protocol

這份診斷是在看過 Experiment 018 的結果之後制定的回溯稽核。它不是 018 之前的預註冊，也不是新的 held-out replication 或盲測。

Protocol SHA-256：`8f13a4e5756a7f6ba712816508e6a0edfc555ee1a4f16b8d1ed0336e367e39e5`

Seeds 700–1699 與 018 的 18 個校準 weight、6 個 weight 150 reference、S0 全部沿用。Weight 從 018 的 weight_hex 讀取。
主要診斷是同一規則在自身閉環歷史與 S0 歷史上、decision t=101–500 的 Q 差。
集中事件是連續 20 個 committed timesteps 至少 18/20 agents 在同一場所。未發生者不記成第 500 步。
對齊錨點是產生該連續區間第一個 committed state 的決策。相對時間是 -40 到 +49，缺歷史不填補。
Q 對 018 Layer A 的容差是 1e-10。HHI、dominant share、entropy 的容差是 1e-12。Replay 浮點容差是 1e-12。機率對 _select 的容差是 1e-12。
Bootstrap 種子是 20261001，10000 次，索引 0 對應 choice seed 700，percentile indexes 249 與 9749。
交叉對照不是 feedback 占比或因果中介。HHI 到頂也不表示記憶沒有終局場所差異。
