# WorldLab V1 Preview 報告

狀態：`V1_PREVIEW`、`REPLAY_ONLY`、`NOT_RELEASED`。

Experiment 021：`NOT_STARTED`。

這是 Experiment 020 既有軌跡的本機回放。V1 Preview 引用 Experiment 020，不取代它，也沒有把 `world_v0_20` 改名。

## 啟動與重建

在 `worldlab/` 執行。

重建展示資料：

```powershell
.\.venv\Scripts\python.exe v1_demo/build_preview.py
```

啟動介面：

```powershell
.\.venv\Scripts\python.exe v1_demo/serve_preview.py
```

瀏覽器打開 `http://127.0.0.1:8765/static/index.html`。

展示資料已經存在時，只需要再執行啟動命令。服務綁定 127.0.0.1，不連外部服務。一般回放只讀取檔案。存取紀錄寫到終端機，不追加到 manifest 所列檔案。

## 展示資料

來源是 Experiment 020 事先指定的 gate seeds：1700、2199、2699。每個 seed 有 18 個 memory × transformation × q 條件、每個條件四條時程，以及一條 S0 donor。合計 219 條。建置程式只讀取這三個 seed 的 `trajectories/*.json` 與 `seeds/*.json`，再用既有的 `unpack_series` 與 `trajectory_hash` 核對。沒有呼叫世界模擬。

late HHI、早期形成、event18 與維持資格沿用種子指標檔裡已經保存的值。當下人數由位置序列在畫面上算出，並標成畫面顯示，不寫回正式統計。

`v1_demo/export/build_record.json` 記錄六個來源檔的 SHA-256、219 條 trajectory hash，以及建置程式 hash。`v1_demo/export/manifest.json` 在內容寫完後產生，不含自己。建置結束時做過唯讀核對。

## 介面修正

比較區在寬度超過 860 CSS pixels 時是左右兩欄，860 以下改成上下排列。場所卡片維持五欄並隨內容增高，20 位代理人都留在卡片裡。

時間滑桿兩端只標 0 與 500。t＝50 的標記在時間範圍的 10%，t＝51 在 10.2%，文字上下錯開，位置不挪開。四個跳轉按鈕放在滑桿外面：初始 t＝0、切換前 t＝50、時程分段 t＝51、結束 t＝500。跳轉會先暫停，再改同一個 `state.t`。整段 late HHI 沿用已保存的摘要。只有 EO 與 OE 會在 decision t＝51 更換來源；左右都是 EE、OO 或 S0 時，頁面寫明它們不會更換。

頁面一開始就停用條件、播放、單步、跳轉、速度與時間滑桿。目錄與三份回放都成功解析，而且第一次畫面畫出之後，才進入可操作狀態。HTTP 失敗、JSON 無法解析或必要內容不存在時，頁面顯示失敗訊息與「重新載入」，控制項維持停用。重試會先清掉播放計時器，不重複綁定事件；較早的載入結果不會覆蓋較新的成功結果。

## 驗證分層

這三項不能互相代替。

### 自動測試

Python 3.12.10 的 `.venv`，這一輪：

- `compileall` 通過。
- `tests/test_v1_preview.py`：3 passed，13.37 秒。其中瀏覽器行為測試用 headless Chrome 操作頁面，失敗案例來自測試伺服器的回應，沒有改寫正式 export 或研究來源。
- `pytest -m "not slow"`：246 passed，7 deselected，156.43 秒。

這一輪沒有重跑完整 slow suite。先前交付時的完整 `pytest` 是 252 passed、1153.82 秒，那個數字不是這一輪的結果。

自動測試檢查了載入前停用、延遲點選、目錄或回放失敗、重試恢復、單一播放計時器、四個跳轉的同步與暫停、late HHI 不隨跳轉改變、1440／1024 並排、768 堆疊，以及切換條件後標題、軌跡與指標一起更新。

### 代理人的瀏覽器操作

代理人用 headless Chrome、`deviceScaleFactor` 1，操作預設 CUM／LOGMAX／q25 的 EE 對 EO，跳到 t＝50、51、500，再改成 EE 對 OE，並切換 memory、transformation、q 與 seed。有事件的 q25、沒有事件的 q05、沒有維持資格的 OE，以及左右都不更換來源的 EE／OO 與 EE／S0，都在同一輪操作裡看過。載入中、載入失敗與重試成功也有操作。

布局斷言使用的 CSS viewport 是 1440×900、1024×900、768×900。這三個寬度都沒有非預期的橫向捲動。1440 與 1024 的兩個比較面板並排，768 改成上下排列。t＝50 時左右都有一個場所放進 20 位代理人，卡片把 00 到 19 都包在裡面。

完整頁面截圖的 CSS 內容寬度與 PNG 像素一致：

- `v1_demo/acceptance_screenshots/desktop_1440.png`：CSS 寬 1440，內容與 PNG 都是 1440×2218。
- `v1_demo/acceptance_screenshots/desktop_1024.png`：CSS 寬 1024，內容與 PNG 都是 1024×2686。
- `v1_demo/acceptance_screenshots/narrow_768.png`：CSS 寬 768，內容與 PNG 都是 768×3403。
- `v1_demo/acceptance_screenshots/load_error.png`：CSS 寬 1024，內容與 PNG 都是 1024×1475。載入失敗訊息與「重新載入」都在圖內。

較早的 `controls.png` 與 `places_and_na.png` 曾被截圖工具裁切，不能當作這次完整布局的證據。

文件與 `v1_demo/README.md` 更新後，已用既有建置程式重寫展示 bundle，並再做一次唯讀 manifest 核對。六個研究來源檔的 hash 與建置前相同。219 條匯出軌跡的 canonical hash 在建置時與來源一致。新的 manifest 列出 13 個檔案且全部吻合，不含自己。

### 尚未進行的目標使用者驗收

目標使用者還沒有自行操作這一頁。上面的自動測試與代理人操作都不能算成這項驗收。狀態保持 `NOT_RELEASED`。

## 與正式 V1 的差距

- 只回放 3 個 seed，不是 1000 seeds 的整體結果。
- 不能由介面產生新世界，也不能重新 fitting。
- 沒有真人行為驗證。
- 場所排列是示意，不表示地理距離。
- 正式 V1 的使用者驗收與發布檢查還沒有做。狀態保持 `NOT_RELEASED`。
