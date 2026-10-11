# V4 Phase 14｜工程 smoke QA

Python 3.12.10，平台 Windows-11-10.0.26200-SP0。角色是 exploratory_development。LLM 呼叫數 0。

讀取路徑只包含：

- `data/curated/v4_forecast_panel_candidate.parquet`
- `data/curated/v4_2024_identity_audit.csv`
- `data/curated/v4_group_comparability.csv`

面板只讀這些欄：
- `group_id`
- `site_uid`
- `year_month`
- `share`
- `dataset_role`

沒有讀取保留檔的人次、份額或 HHI。合成別名 site_01 對到 stat_ 識別碼時會被拒絕。 這次檢查是 True。相同 seed 重跑是 True。

開發月是各群第一個事前可評估月，用來檢查截止日與基準介面。它不是 P4，也不用來改測試月或參數。

| 地理群 | 可評估月 | 開發月 | 情境 A 最後來源 | 情境 B 最後來源 | ABM | 2025 單步 |
| --- | ---: | --- | --- | --- | --- | --- |
| 南投縣／玉山國家公園 | 96 | 2014-01 | 2013-12 | 2013-10 | prediction_unavailable | unavailable_incomplete_2024_identity |
| 花蓮縣／太魯閣國家公園 | 120 | 2014-01 | 2013-12 | 2013-10 | prediction_unavailable | unavailable_incomplete_2024_identity |
| 苗栗縣／雪霸國家公園 | 120 | 2014-01 | 2013-12 | 2013-10 | prediction_unavailable | unavailable_incomplete_2024_identity |

七個基準在這個開發月都回傳 predicted。ABM 通道因為真實屬性不足而 prediction_unavailable。預測沒有寫回觀測歷史。

受保護檔案在寫出報告前後的雜湊相同。
