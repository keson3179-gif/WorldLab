# V4 Phase 15｜真實場所卡

這一輪是 exploratory development。沒有執行 P4，沒有呼叫 LLM，也沒有讀取 2025–2026 保留人次。九處據點來自 Phase 14 已固定的三個地理群，不是依預測誤差挑選。

景點資料庫檔案更新時間是 2026-10-10T14:30:04+08:00。同名同縣市的欄位記為 source_reported，沒有升成 verified。模糊名稱只出現在可達性候選表。

| 據點 | 縣市 | 座標配對 | 統計類別 | 2024 身分狀態 |
| --- | --- | --- | --- | --- |
| 塔塔加遊憩區 | 南投縣 | 無同名列 | 國家公園 | rule_inferred_not_confirmed |
| 排雲山莊 | 南投縣 | 無同名列 | 國家公園 | not_found |
| 玉山管理處遊客服務中心 | 南投縣 | 無同名列 | 國家公園 | rule_inferred_not_confirmed |
| 太魯閣國家公園遊客中心 | 花蓮縣 | 無同名列 | 國家公園 | rule_inferred_not_confirmed |
| 布洛灣遊憩區 | 花蓮縣 | exact_name_county | 國家公園 | rule_inferred_not_confirmed |
| 臺八線沿線景觀區 | 花蓮縣 | 無同名列 | 國家公園 | rule_inferred_not_confirmed |
| 汶水遊客中心 | 苗栗縣 | 無同名列 | 國家公園 | rule_inferred_not_confirmed |
| 觀霧遊客中心 | 苗栗縣 | 無同名列 | 國家公園 | rule_inferred |
| 雪見遊憩區 | 苗栗縣 | exact_name_county | 國家公園 | rule_inferred |

觀霧與雪見的 safe_to_link 只是 CSV 類型未變的規則推定，human_reviewed 仍是 false。雪見的 Excel 合併類型用字和 CSV 不同。人次相同沒有被當成口徑連續。

模糊候選沒有寫入場所卡座標：
- 太魯閣國家公園遊客中心 對上「太魯閣牌樓」（花蓮縣）。
- 太魯閣國家公園遊客中心 對上「太魯閣大橋」（花蓮縣）。
- 太魯閣國家公園遊客中心 對上「太魯閣國家公園」（花蓮縣）。
- 太魯閣國家公園遊客中心 對上「太魯閣臺地」（花蓮縣）。
- 布洛灣遊憩區 對上「布洛灣吊橋」（花蓮縣）。
- 汶水遊客中心 對上「汶水老街」（苗栗縣）。
- 觀霧遊客中心 對上「觀霧國家森林遊樂區」（苗栗縣）。
