# WorldLab V4 QA 摘要

PASS 9、WARN 4、FAIL 0。FAIL 會讓 `qa` 指令以非零狀態結束。

| 階段 | 檢查 | 結果 | 說明 |
| --- | --- | --- | --- |
| schema | pandera visit schema | PASS | rows=49512 |
| duplicates | one row per site and month | PASS | 0 |
| missing | missing is not stored as zero | PASS | 0 |
| range | no negative reported visits | PASS | 0 |
| annual | month sum versus published total | WARN | unreconciled_site_years=317; month values were not rewritten |
| site_id | attraction master site_id unique | PASS | 0 |
| crosswalk | fuzzy score did not confirm a site | PASS | 0 |
| groups | a site is in at most one included group | PASS | 0 |
| groups | official backtest groups | WARN | eligible_rows=0 |
| leakage | strict features require known availability | PASS | 0 |
| versions | conflicting visit versions stay out of the panel | PASS | 0 |
| publication | visitor publication history | WARN | publication_history_unknown |
| measurement | counting method | WARN | measurement_method=unknown，未推定各據點相同 |

pytest：`compileall` 通過。`pytest -m "not slow"` 為 259 passed、7 deselected，93.88 秒。V4 資料測試 13 項包含在其中。沒有重跑 slow 全套。
