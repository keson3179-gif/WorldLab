# V4 Phase 19｜前瞻封存稽核

Phase 18 的 ProspectiveRegistry 是合成資料上的原型。它能拒絕覆寫既有 snapshot_id，也會在修訂後比對第一版雜湊。它還不能當作前瞻預測的封存系統。本輪只改這個原型與它的測試，沒有下載或讀取 2025–2026 的保留人次、份額或 HHI。

本機檔案系統不是不可竄改的儲存。擁有檔案權限的人仍可改磁碟上的位元組。雜湊核對可以發現內容與清單不一致，不能證明修改發生在哪個外部時間。若要證明預測早於目標資料公開，必須把清單雜湊交到註冊庫之外的時間戳記。

## 原型的缺口

1. 凍結之後、封存觀察之前，`replace_prediction` 可以改寫預測雜湊。
2. `seal_observation` 直接改寫同一份 `forecast.json`。
3. 凍結只收下呼叫端宣稱的雜湊，沒有保存預測檔、設定與程式版本的實際位元組。
4. 時間是字串比較，沒有時區的時間也會被接受。
5. 只有 `frozen_at` 與 `available_for_forecast_at`，沒有分開輸入截止、凍結、目標首次公開與首次取得。
6. 沒有保存首次取得的原始觀察位元組。
7. 觀察沒有追加修訂，第一次紀錄會被後來的寫入改掉。
8. 預測檔寫入後沒有強制把磁碟內容重算雜湊。
9. `snapshot_id` 直接成為路徑，`../` 可以跳出登錄目錄。
10. 沒有中斷寫入與重複封存的測試。

## 修正後的契約

`worldlab_observation/external_evidence.py` 的 ProspectiveRegistry 現在遵守下列規則。

- 預測一旦 `freeze_forecast` 完成，`replace_prediction` 一律拒絕。修訂必須使用新的 `forecast_id`，並在清單寫上 `revises`。前一版的清單與預測位元組保持不變。
- 凍結時寫入 `prediction.bin`、`config.bin`、`code_version.bin`，清單保存三者的 sha256。`verify_forecast` 重讀磁碟後比對。
- `input_cutoff_at`、`frozen_at`、`target_first_published_at`、`first_retrieved_at` 必須是帶時區的時間，並正規化成 UTC。無時區的字串會被拒絕。合法順序是輸入截止不大於凍結時間，而且凍結時間早於目標首次公開。首次取得不得早於目標首次公開。
- `record_first_observation` 把第一份原始觀察寫到另一個目錄，不改凍結清單。第二次封存同一預測會失敗。
- 官方資料日後改版時，`append_observation_revision` 追加一筆修訂。第一份 `content.bin` 不變。
- 快照與預測寫入時先留下 `.incomplete`。清單寫入中斷時，`verify_snapshot` 不把該目錄當成有效紀錄，也不能用同一個識別碼補寫成功。
- 識別碼只允許字母、數字、點、底線與連字號。路徑跳脫會在建立目錄前被拒絕。
- 清單中的 `storage_guarantee` 固定為 `api_append_only_not_tamper_proof`。

## 外部時間先後

本地登錄只保證這個 API 不覆寫自己的完成紀錄。它不保證外人無法改檔，也不保證凍結時間就是牆鐘上的那個時刻。

若未來要做 P4b，應在 `target_first_published_at` 之前，把 `forecast_id`、`prediction_sha256`、`config_sha256`、`code_version_sha256`、`input_cutoff_at` 與 `frozen_at` 交給下列其中一項：

- RFC 3161 時間戳記服務，保存回傳的時間戳記權杖。
- 公開的研究註冊，例如在目標資料公開前記下同一組雜湊與目標月份。

本輪沒有呼叫時間戳記服務，也沒有建立註冊帳號。沒有這份外部紀錄時，不能把本地凍結時間說成已經被獨立證明。
