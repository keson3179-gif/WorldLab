# V4 Phase 18｜前瞻快照規格

這份規格用合成內容測試，沒有對現有 2025–2026 保留檔取用。原始快照寫入後不可覆寫。資料日後修訂時另開 snapshot_id，第一版的 content_sha256 保持不變。

每次官方資料更新要留下 snapshot_id、retrieved_at、source_url、http_metadata、content_sha256、data_version、latest_observation_month、available_for_forecast_at、forecast_origin、model_version、forecast_manifest_hash、prediction_file_hash。

預測的 frozen_at 必須早於目標資料的 available_for_forecast_at。觀察結果寫入之後，同一版預測不能再改 prediction_file_hash。多個事前凍結的模型可以並存。API 失敗、資料延遲與缺報各寫一筆不可覆寫的失敗紀錄。

實作在 `worldlab_observation/external_evidence.py` 的 ProspectiveRegistry。它現在只證明這些不變量，還沒有接上真實的觀光署更新。
