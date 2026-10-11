# Experiment 018 protocol

狀態：校準前固定。本文件不含校準結果，也不含 seeds 700–1699 的世界結果。

Protocol SHA-256：`78676f967081b3c964e70c839afbe47991c4e913dd6cbc2e6cfedc9be8d74b7c`

Calibration cohort 是 Experiment 017 已保存的 S0，seeds 200–699。Held-out cohort 是 seeds 700–1699。
Q targets 固定為 0.05、0.15、0.25。校準容差是 0.0005。Held-out 平均匹配帶是 target ± 0.005，配對 Q 差的帶是 [-0.005, 0.005]。
Solver 從 weight 0 與上界 150 開始，必要時把上界倍增，上限 1,000,000，之後做最多 64 次二分。搜尋上限沒有心理學含義。
主要閉環結果是 t=101–500 累積 visit share 的 HHI。主要對照是九個 D_matched、三個 D_150，以及 D_matched − D_150。
Bootstrap 使用分析種子 20260930 的本地 Random，10000 次，每列 1000 個有放回索引，percentile indexes 249 與 9749。索引 0 是 choice seed 700。
這個區間給的是已固定 weights 之下新 seeds 的不確定性，不包含重新 fitting 的校準不確定性。

平均 Q 相同，不代表選擇機率向量、時間分布、agent 分布或閉環影響相同。
這不是純尺度識別、純記憶時間尺度、真人行為校準、路徑依賴或外部效度。
Held-out 匹配帶沒有通過時，仍依固定 weights 完成分析，不重新 fitting。

計數：lineage 371、新世界 25000、replay 75、合計 25446。Calibration 不新增世界。Layer A 不另計 world。
