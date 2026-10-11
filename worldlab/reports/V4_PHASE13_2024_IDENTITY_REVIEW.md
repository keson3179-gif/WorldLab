# V4 Phase 13｜2024 高證據身分覆核

這份覆核與探索性基準分開。它沒有改四組歷史回測樣本，也沒有把任何列升成 confirmed_manual。沒有真人逐項審核，human_reviewed 維持 false。

Phase 12 的 24 列裡，unresolved 是 18、unchanged 是 4、not_found 是 2。這 13 列是 unresolved 之中、2024 年十二個月人次與同年 Excel 同名列一致的高優先候選。

兩個判斷分開記。physical_or_statistical_identity_continuity 這裡記成 cross_source_evidence_candidate，表示同年 CSV 與 Excel 可能是同一統計列。measurement_continuity 記成 not_established。十二個月人次一致不能單獨證明與上一年的計數方式或統計範圍連續。

| 據點 | 縣市 | CSV 2023 | CSV 2024 | Excel 2024 | 方法 2023 | 方法 2024 | 覆核狀態 | 口徑連續 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 塔塔加遊憩區 | 南投縣 | 國家公園 | 其他 | 國家(自然)公園 | 電子計數器 | 電子計數器 | cross_source_evidence_candidate | not_established |
| 觸口遊客中心 | 嘉義縣 | 國家級風景特定區 | 其他 | 國家級風景特定區 | 電子計數器 | 電子計數器 | cross_source_evidence_candidate | not_established |
| 梅山太平雲梯 | 嘉義縣 | 國家級風景特定區 | 宗教場所 | 國家級風景特定區 | 門票數 | 門票數 | cross_source_evidence_candidate | not_established |
| 達娜伊谷 | 嘉義縣 | 國家級風景特定區 | 其他 | 國家級風景特定區 | 門票數 | 門票數 | cross_source_evidence_candidate | not_established |
| 太魯閣國家公園遊客中心 | 花蓮縣 | 國家公園 | 國家級風景特定區 | 國家(自然)公園 | AI精密人流偵測 | AI精密人流偵測 | cross_source_evidence_candidate | not_established |
| 布洛灣遊憩區 | 花蓮縣 | 國家公園 | 國家級風景特定區 | 國家(自然)公園 | AI精密人流偵測 | AI精密人流偵測 | cross_source_evidence_candidate | not_established |
| 臺八線沿線景觀區 | 花蓮縣 | 國家公園 | 其他 | 國家(自然)公園 | AI精密人流偵測 | AI精密人流偵測 | cross_source_evidence_candidate | not_established |
| 汶水遊客中心 | 苗栗縣 | 國家公園 | 直轄市及縣(市)級風景特定區 | 國家(自然)公園 | 電子計數器 | 電子計數器 | cross_source_evidence_candidate | not_established |
| 翟山坑道 | 金門縣 | 國家公園 | 其他 | 國家(自然)公園 | 電子計數器 | 電子計數器 | cross_source_evidence_candidate | not_established |
| 中山林遊客中心 | 金門縣 | 國家公園 | 國家級風景特定區 | 國家(自然)公園 | 電子計數器 | 電子計數器 | cross_source_evidence_candidate | not_established |
| 湖井頭戰史館 | 金門縣 | 國家公園 | 其他 | 國家(自然)公園 | 電子計數器 | 電子計數器 | cross_source_evidence_candidate | not_established |
| 蔣經國先生紀念館 | 金門縣 | 國家公園 | 其他 | 國家(自然)公園 | 電子計數器 | 電子計數器 | cross_source_evidence_candidate | not_established |
| 古寧頭戰史館 | 金門縣 | 國家公園 | 國家級風景特定區 | 國家(自然)公園 | 電子計數器 | 電子計數器 | cross_source_evidence_candidate | not_established |

13 列的 review_status 都是 cross_source_evidence_candidate。confirmed_manual 都是 false。baseline_sample_changed 都是 false。

地方景點清單 AttractionList.xml 以名稱完全相符查到：布洛灣遊憩區、翟山坑道、湖井頭戰史館、古寧頭戰史館。名稱出現只表示清單上有這個地名，沒有記載 2023 到 2024 的拆分、合併或統計範圍變動。其餘據點沒有同名景點列，這也不構成範圍已經改變的證據。

CSV 類型與 Excel 合併類型的差異仍保留。同名不足以接上 2023 的 site_uid。
