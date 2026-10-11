# Experiment 020 更正

這是既有 Experiment 020 結果的文件更正與唯讀封存稽核。它不是 Experiment 021，沒有新增正式世界，沒有重新校準權重，也沒有重算正式 bootstrap。

封存報告是 `data/worlds/world_v0_20/EXPERIMENT_020_REPORT.md`。本輪沒有改寫它。更正後的解釋在 `docs/EXPERIMENT_020_REPORT.md`。稽核程式、結果與來源清單在 `docs/experiment_020_audit/`。

稽核開始時，兩份報告的文字相同，都含有下面被更正的句子。那是執行結束後補上的說明，並且曾被複製進 data 報告。這次稽核把 data 那份留在原處，只改正 docs 報告。因此兩份報告現在的差異，就是本文件列出的解釋更正。

## 退出時間

EO 的 qualified 世界並非每一個都在 t=51 解除集中。六個 LOGMAX q15/q25 條件的 `original_exit` 與 `release_time` 分布相同，都是 t=51 與 t=52：

| condition | eligible | t=51 | t=52 | maintained | released | transfer |
| --- | --- | --- | --- | --- | --- | --- |
| CUM LOGMAX q15 | 979 | 978 | 1 | 0 | 979 | 0 |
| EXP99 LOGMAX q15 | 942 | 941 | 1 | 0 | 942 | 0 |
| RW50 LOGMAX q15 | 782 | 781 | 1 | 0 | 782 | 0 |
| CUM LOGMAX q25 | 1000 | 999 | 1 | 0 | 1000 | 0 |
| EXP99 LOGMAX q25 | 1000 | 999 | 1 | 0 | 1000 | 0 |
| RW50 LOGMAX q25 | 998 | 997 | 1 | 0 | 998 | 0 |

正確敘述是：所有合格世界都在 t=51–52 解除集中，退出時間的中位數為 t=51。分母、維持率、風險差與既有信賴區間不變。完整次數來自已保存的 seed metrics，記在 `audit_results.json` 的 `maintenance`。

## EO−OO 區間

`early_given_O` 共 18 格，讀自凍結的 `uncertainty_analysis.json`，沒有重算。17 格的區間包含 0。唯一不包含 0 的是 EXP99 / ZRAW / q25：

- mean = 1.3206593749999967e-05
- CI = [2.0166250000004103e-06, 2.4608343750000926e-05]

效應幅度很小，而且區間完全高於 0。本輪沒有新增等價判準。接近零，或區間包含 0，都不是已證明沒有差異。

## used–own field gap

`world/experiment_020.py` 的計算是 `mean(abs(used_values - own_values))`。E 把 `used` 設成 recipient 的 own score，所以這個欄位為 0。O 把 `used` 設成 donor cube，所以只有 O 才是 donor 與 own 的差。它不能用來描述 E 下 donor 與 own 是否相同。

凍結 JSON 的欄位名稱仍是 `field_gap_t1_50`、`field_gap_t51_100`、`field_gap_t101_500`、`field_gap_t1_500`。文件中的正確稱呼是 used–own field gap，也就是實際使用場與自身內生場的差距。數值與欄位名稱都沒有改。

## manifest 封存順序

`run_manifest.json` 列出 2020 個檔案。以 1 MiB 串流 SHA-256 核對後，2019 個符合，整體 `manifest_all_matched` 是 false。唯一不符的是 `progress.jsonl`。

manifest 記錄的 hash 是 `4274375e426f810ee39a3265cd6212a43cfe3f5f9dffcb6f8c44921a648ab643`。目前完整檔的 hash 是 `50a467909d67083db05599947c04d88bf2697aa4a6f21e1aa7d6186612983599`。去掉最後一筆之後的原始 bytes 共 9682 bytes，其 hash 等於 manifest 記錄。剩餘 92 bytes 恰好是一筆換行結尾的 JSONL，`stage` 為 `finished`，`status` 為 `COMPLETE`，`elapsed_seconds` 為 6102.0948699999135。

這可以解釋為：執行器先寫 manifest，之後才追加 finished 紀錄。其餘被列出的檔案都吻合。已解釋的差異仍然使整份 manifest 不能標成全部吻合。原始 manifest、progress 與 data 報告都保留。

未來執行器應先完成結果與最後一筆進度紀錄，再產生 manifest，再做唯讀驗證。manifest 建立之後，不再追加它所涵蓋的檔案。這項要求只記在這裡，沒有修改已納入來源 hash 的 `experiment_020.py`。

## 退化區間與耗時

[0, 0] 或 [1, 1] 可以來自這 1000 個 seeds 的觀測值完全相同。它不表示未見 seeds 的母體機率也必然是 0 或 1。既有區間保留。

三個耗時各有來源，凍結數值都沒有改：

- `completion_status.json` 的 `elapsed_seconds` 是 6096.137776999967，寫在圖、報告與 manifest 之前。
- 封存報告開頭的 6098.0 秒，是其後寫入報告時取到小數一位的經過時間。
- `progress.jsonl` 最後一筆 finished 紀錄是 6102.0948699999135 秒。終端的 6102.1 秒是這個數取到小數一位。

若稱正式執行器總耗時，採用這筆完成紀錄。

## 主要結論

這些更正保留主要方向。在已保存的結果裡，高權重 LOGMAX 的早期內生來源仍對應較高的早期形成比例；切換到外生來源後，合格世界在 t=51–52 解除集中，中位數是 t=51。早期外生、後期恢復內生時，LOGMAX q25 仍有晚期強集中。EXP99 ZRAW q25 的 EO−OO 區間高於 0，幅度約 10⁻⁵，不改變這個集中結論，也不能被說成已證明沒有差異。
