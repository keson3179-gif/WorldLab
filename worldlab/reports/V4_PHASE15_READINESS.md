# V4 Phase 15｜就緒狀態

Python 3.12.10，平台 Windows-11-10.0.26200-SP0。LLM 呼叫數 0。

TDX 狀態是 offline_credentials_absent。dry_run_plan 的 will_call_network 是 False。沒有呼叫交通 API。直線距離也沒有改稱為旅行時間。現況交通供給不能回填 2014 年。

模糊候選列數 7。同群之內沒有兩處同名同縣市座標，所以沒有場所間直線距離矩陣。

2024 身分沒有因為 CSV 與 Excel 人次相同而升成口徑連續。三群的 2025 單步資格：
- 南投縣／玉山國家公園：unavailable_incomplete_2024_identity。
- 花蓮縣／太魯閣國家公園：unavailable_incomplete_2024_identity。
- 苗栗縣／雪霸國家公園：unavailable_incomplete_2024_identity。

備援方案是前瞻快照，不是改截止日。若要讓未來某個月能做公平單步預測，必須在該月分數被計算之前，凍結據點集合、場所卡版本、資訊情境與主指標 share_vector_TV。2024 身分未完整時，不能拿 2023-12 當作 t-1。本輪沒有凍結確認性協定，也沒有產生真實場所的探索性份額預測。

下一輪還不能正式比較 ABM 與七個統計基準的真實景點分數。比較框架可以保留；M0 與 M1 在真實場所上仍是 prediction_unavailable。

讀取路徑：

- `data/curated/v4_forecast_panel_candidate.parquet`
- `data/curated/v4_2024_identity_audit.csv`
- `data/curated/v4_group_comparability.csv`
- `data/curated/v4_canonical_site_crosswalk.csv`
- `觀光資料標準V2.1/AttractionList.xml`

景點資料庫 SHA-256：`52b04217c73814d963a46c7a0f225e7a1004c10f709b2fd24e445275e479b0cd`。
