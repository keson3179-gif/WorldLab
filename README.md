# WorldLab 公開審查材料

這個儲存庫提供國科會審查用的方法、程式與前導證據。正式計畫書另送國科會系統，不在這裡。

材料對應的研究是：生成式人工智慧輔助的代理人基礎模型，用來檢查觀光場所的社會資訊回饋、歷史優勢與集中動態在什麼條件下形成，以及這些條件的適用界線。這裡的結果不表示模型已經能預測真實旅客行為。

## 建議閱讀順序

1. [`worldlab/docs/RESEARCH_SYNTHESIS_V020.md`](worldlab/docs/RESEARCH_SYNTHESIS_V020.md)  
   Experiment 001–020 的模型內證據與不能外推的範圍。Experiment 021 尚未開始。
2. [`worldlab/reports/V4_PHASE19_FINAL_DECISION.md`](worldlab/reports/V4_PHASE19_FINAL_DECISION.md)  
   外部檢驗目前停在哪裡。
3. [`worldlab/reports/V4_PHASE17_GO_NO_GO.md`](worldlab/reports/V4_PHASE17_GO_NO_GO.md) 與 [`worldlab/reports/V4_PHASE18_GO_NO_GO.md`](worldlab/reports/V4_PHASE18_GO_NO_GO.md)  
   原三群 M0+X、原九處歷史網路關注，兩項可行性驗收都未通過。
4. [`worldlab/reports/V4_PHASE19_P4_REDESIGN.md`](worldlab/reports/V4_PHASE19_P4_REDESIGN.md)  
   P4a 與 P4b 的區分。P4a 尚未計算。P4b 尚未執行，也不能用 P4a 代替。

程式分成兩段。`worldlab/world/` 是凍結效用之後的 Python 世界模擬。`worldlab/worldlab_observation/` 與 `worldlab/worldlab_data/` 是官方人次的稽核、基準比較與前瞻封存契約。

## 收錄範圍

納入的是原始碼、測試、實驗協議、Phase 8–19 報告，以及身分、計數標籤、特徵資格這類小型稽核表。`worldlab/v1_demo/` 是模型內重播預覽，狀態為尚未正式發布，不是真實旅客預測。

沒有納入計畫書、政府原始檔、月人次明細、2025–2026 保留集、模擬軌跡 JSONL，以及金鑰。完整對照見 [`PUBLIC_SCOPE.md`](PUBLIC_SCOPE.md)。

## 本機執行

需要 Python 3.12。在 `worldlab/` 建立虛擬環境後安裝 `requirements-lock.txt`。快速測試：

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not slow"
```

部分資料測試會因原始政府檔不在這個儲存庫而跳過。請不要把保留月份的人次補進儲存庫，也不要依那些月份的誤差挑選據點或模型。`.env.example` 只有空的欄位名稱。
