# 公開範圍

這個目錄就是公開儲存庫的根目錄。請上傳這裡的內容，不要上傳上一層的研究工作目錄。

## 納入

| 路徑 | 內容 |
| --- | --- |
| `worldlab/world/`、`agents/`、`experiments/`、`llm/`、`models/` | 模型內模擬與效用上游 |
| `worldlab/worldlab_data/`、`worldlab_observation/` | 觀察資料稽核與封存契約 |
| `worldlab/tests/` | 測試。前瞻封存測試使用合成內容 |
| `worldlab/docs/` | Experiment 001–020 協議、報告與綜合 |
| `worldlab/reports/` | Phase 8–19 與資料品質報告 |
| `worldlab/configs/` | 資料路徑設定與人工交叉表。路徑指向未公開的原始檔 |
| `worldlab/v1_demo/` | 三個模型內重播，不是官方人次預測 |
| `worldlab/data/curated/` 的稽核表 | 2021 對齊、2024 身分、據點連續性、特徵資格、方法標籤史 |
| `worldlab/data/v4_observational/` 的摘要 | Phase 摘要、身分缺口、頁面比對。沒有月人次明細 |

計數方法史只保留工作簿上的方法標籤與月份數，不放逐月人次。政府資料的公開入口是 [data.gov.tw 資料集 8116](https://data.gov.tw/dataset/8116)。

## 未納入

- 正式計畫書 Word、PDF 與圖。這些另送國科會系統。
- `政府公開資料/` 的原始 CSV、Excel、氣象、活動與行事曆檔。
- 2025–2026 保留人次、份額與 HHI。程式裡的 `holdout_candidate` 是分隔規則，不是那些數字。
- 探索性基準分數、月人次面板與 `month_scores.csv`。
- `data/` 內的 JSONL、Parquet 與世界軌跡。
- 虛擬環境、`.env`、編輯器設定與代理程式說明。

檔案清單見 `FILE_LIST.txt`。`worldlab/` 內有 331 個檔，約 13.5 MB。
