# V4 Phase 17｜是否進入 M0 + X

Python 3.12.10，平台 Windows-11-10.0.26200-SP0。決策是 NO-GO。LLM 呼叫數 0。

沒有一項特徵同時滿足：預測原點前可取得、三個據點都有值、而且同月的值在群內不同。週末與放假日在嚴格意義上週末可事前知道，但群內沒有差異。氣象只有部分據點的事後或發布日不明的代理。社會資訊沒有獨立來源。

因此下一輪不開發 M0 + X 的份額回測，也不把缺值填起來、擴大 beta 搜尋或呼叫 LLM。若要改變這個決定，需要新的、帶發布日、而且能在同一群三個據點之間分開的歷史資料。

仍不能由現有觀察資料識別的命題：旅客個人偏好、社會曝光的因果效果、Q、E/O 來源替換，以及 2024 以後的公平單步預測。

讀取路徑：

- `data/curated/v4_forecast_panel_candidate.parquet`
- `data/curated/v4_group_comparability.csv`
- `data/curated/v4_canonical_site_crosswalk.csv`
- `data/curated/v4_2024_identity_audit.csv`
- `data/curated/v4_2024_priority_identity_review.csv`
- `data/curated/v4_real_place_atlas.csv`
- `政府行事曆`
- `氣象資料/02-pipeline/data_processed/station_month_heat_metrics_v0.2.csv`
- `氣象資料/02-pipeline/metadata/attraction_station_primary_high_v0.2.csv`
- `氣象資料/02-pipeline/metadata/station_registry.csv`
- `活動 - 觀光資訊資料庫/EventList.xml`
