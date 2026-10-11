# WorldLab V1 Preview

本機回放 Experiment 020 已保存的三個 gate seeds。狀態是 `V1_PREVIEW`、`REPLAY_ONLY`、`NOT_RELEASED`。

在 `worldlab/` 執行。

重建展示資料：

```powershell
.\.venv\Scripts\python.exe v1_demo/build_preview.py
```

啟動介面（只綁定 127.0.0.1）：

```powershell
.\.venv\Scripts\python.exe v1_demo/serve_preview.py
```

瀏覽器打開 `http://127.0.0.1:8765/static/index.html`。

桌面寬度下左右視窗並排，窄視窗改成上下排列。時間滑桿兩端是 0 與 500；t＝50 與 t＝51 的跳轉按鈕不是比例刻度。回放資料載入完成前，頁面上的回放操作是停用的。載入失敗時按「重新載入」。

展示資料已經建好時，只要再執行啟動命令，不必重建。服務不會寫入 `data/`，也不會把存取紀錄追加進 `v1_demo/export/manifest.json` 所列出的檔案。
