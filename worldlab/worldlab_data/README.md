# WorldLab V4 觀察式資料

這個目錄只處理觀光公開資料的盤點、清洗、對齊與品質驗收。它不跑 WorldLab 模擬，不呼叫 LLM，也不改 Experiment 001–020 的結果。

原始檔放在設定檔 `configs/v4_data.json` 的 `raw_root`，預設是工作區的 `政府公開資料`。程式只讀那個目錄，輸出寫到 `data/v4_observational/` 與 `reports/`。

## 安裝

在 `worldlab` 使用既有的 Python 3.12 虛擬環境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

空間套件 geopandas 這輪沒有安裝。測站距離沿用既有研究裡已經算好的距離，不在這裡重算。

TDX 金鑰若之後要在本機使用，複製 `.env.example` 為 `.env`。`.env` 不要提交。

## 執行順序

```powershell
.\.venv\Scripts\python.exe -m worldlab_data inventory
.\.venv\Scripts\python.exe -m worldlab_data clean
.\.venv\Scripts\python.exe -m worldlab_data match-sites
.\.venv\Scripts\python.exe -m worldlab_data build-panel
.\.venv\Scripts\python.exe -m worldlab_data qa
.\.venv\Scripts\python.exe -m worldlab_data tdx-fetch --dry-run
```

每個指令都可以加 `--dry-run`。dry-run 只計算並印出摘要，不寫輸出檔。`tdx-fetch` 在這輪不會連網；沒有金鑰時也只印出離線計畫。

## 輸出

- `reports/data_inventory.md`：檔案盤點
- `reports/source_field_mapping.csv`：來源欄名
- `reports/missing_data_requirements.md`：缺項
- `data/v4_observational/processed/site_month_visits.parquet`：月別人次長表
- `data/v4_observational/processed/attraction_master.parquet`：統計據點主檔
- `data/v4_observational/curated/site_crosswalk.csv`：配對表，未確認的列不是最終對照
- `reports/site_matching_review.csv`：待人工覆核的候選
- `data/v4_observational/processed/site_month_panel.parquet`：正式回測面板。計數方式或發布時間未確認時，這份是空的
- `data/v4_observational/processed/exploratory_visit_share_panel.parquet`：探索用份額，不能當成回測結果

## 例外與重跑

缺值不會填 0。同一據點月份若兩個來源數字不同，兩筆都留在隔離檔，不挑一個。模糊比對不會覆蓋 `configs/v4_site_crosswalk_manual.csv` 裡 `confirmed_manual` 的列。

重跑時再執行上面的指令即可。輸出會被重新產生；原始目錄不會被改寫。`qa` 若有 FAIL，行程式結束代碼是 1。
