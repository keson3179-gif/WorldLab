# 重製 Research Synthesis v0.16 的圖

在 `worldlab/` 執行。Python 版本是 3.12；本輪圖使用 CPython 3.12.10。

先安裝核心環境，再安裝繪圖依賴。繪圖檔是 `docs/research_synthesis_v016/plotting-requirements.txt`，其中固定 `matplotlib==3.11.2` 與當時使用的繪圖套件。不要用它取代 `requirements-lock.txt`。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r docs/research_synthesis_v016/plotting-requirements.txt
.\.venv\Scripts\python.exe docs/research_synthesis_v016/build_synthesis.py
```

SVG 與 PNG 都需要這個繪圖環境。只重算衍生表與證據索引時，程式可以在沒有 matplotlib 的環境載入；一旦實際畫圖而套件不在，程式會印出上面的安裝指令並停止，不會自行安裝。

中文圖使用 Windows 內建的 Microsoft JhengHei。實際選用的字型寫在 `source_manifest.json` 的 `font`。沒有可用的正體中文字型時，重建會停止。

這項重建只讀取已凍結的分析檔，並寫入 `docs/research_synthesis_v016/`。不需要、也不會重跑任何正式世界實驗。
