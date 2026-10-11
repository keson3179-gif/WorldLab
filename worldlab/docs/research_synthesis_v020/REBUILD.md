# 重製 Research Synthesis v0.20 的表與圖

在 `worldlab/` 執行。Python 版本是 3.12；本輪圖使用 CPython 3.12.10。

先安裝核心環境，再安裝繪圖依賴。繪圖檔是 `docs/research_synthesis_v020/plotting-requirements.txt`，其中固定 `matplotlib==3.11.2` 與當時使用的繪圖套件。不要用它取代 `requirements-lock.txt`，也不要改 `docs/research_synthesis_v016/` 的檔案。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r docs/research_synthesis_v020/plotting-requirements.txt
.\.venv\Scripts\python.exe docs/research_synthesis_v020/build_synthesis.py
```

資料擷取與 evidence 檢查可以在沒有 matplotlib 的環境載入。只有實際畫圖時才載入繪圖套件。套件不在時，程式會印出上面的安裝指令並停止，不會自行安裝。

中文圖使用 Windows 內建的 Microsoft JhengHei。實際選用的字型、Python 與 matplotlib 版本寫在 `source_manifest.json`。沒有可用的正體中文字型時，重建會停止。

這項重建只讀取本輪列出的凍結摘要、正式 uncertainty 與 Experiment 020 稽核結果，並寫入 `docs/research_synthesis_v020/` 與 `docs/RESEARCH_SYNTHESIS_V020.md` 的雜湊紀錄。不重跑世界、不重新 fitting、不重算正式 bootstrap，也不呼叫 LLM。`source_manifest.json` 不包含自己的 hash。
