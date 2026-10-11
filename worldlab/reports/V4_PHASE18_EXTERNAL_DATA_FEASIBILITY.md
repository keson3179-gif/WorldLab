# V4 Phase 18｜外部資料可行性

查核時間 2026-10-10T17:48:00Z。瀏覽量窗口固定為 2015-07:2023-12，沒有讀取 2025–2026 的人次、份額或 HHI。Phase 17 的 NO-GO 沒有改寫。

中文維基條目必須指向該據點。國家公園總頁不當作園內遊客中心或沿線景觀。瀏覽量是網路關注代理，不是旅客實際曝光，也不是遊客人次。

| 據點 | 條目 | 證據 | 最早月份 | 覆蓋率 | 可用 |
| --- | --- | --- | --- | --- | --- |
| 塔塔加遊憩區 | 塔塔加 | exact_title_redirect | 2015-07 | 1.000 | true |
| 排雲山莊 | 排雲山莊 | exact_title | 2015-07 | 1.000 | true |
| 玉山管理處遊客服務中心 | — | no_unique_article | — | 0.000 | false |
| 太魯閣國家公園遊客中心 | — | no_unique_article | — | 0.000 | false |
| 布洛灣遊憩區 | 布洛灣 | stem_title_with_geography | 2015-11 | 0.412 | false |
| 臺八線沿線景觀區 | — | no_unique_article | — | 0.000 | false |
| 汶水遊客中心 | — | no_unique_article | — | 0.000 | false |
| 觀霧遊客中心 | — | no_unique_article | — | 0.000 | false |
| 雪見遊憩區 | — | no_unique_article | — | 0.000 | false |

原九處裡，覆蓋率達到 0.80 且至少 36 個月的條目有 2。名稱對上、但覆蓋不足的不計入。同一地理群有兩條可用序列的是：玉山國家公園。玉山仍缺第三處條目，所以原本的 J=3 沒有補齊。同名重新導向只表示百科使用該標題，條目範圍可能廣於統計據點。

Wikimedia Commons 地理影像與 OSM 歷史版本沒有本地、帶月份的授權底本。照片上傳量不是遊客人次，OSM 編輯日期也不是設施開放日。這兩項是 NO-GO，本輪不建立擷取工程。

若原九處不能形成對照，候選清單只按 2015 年以前已有統計、且連續到 2023 年、名稱不是遊客中心或管理處、每縣一處來選。沒有使用預測誤差。

| 候選 | 縣市 | 條目 | 證據 | 覆蓋率 | 可用 |
| --- | --- | --- | --- | --- | --- |
| 日月潭環潭區 | 南投縣 | — | no_unique_article | 0.000 | false |
| 圓潭自然生態園區 | 嘉義縣 | — | no_unique_article | 0.000 | false |
| 和平島公園 | 基隆市 | — | no_unique_article | 0.000 | false |
| 蘇澳冷泉 | 宜蘭縣 | 蘇澳冷泉 | exact_title | 1.000 | true |
| 佳樂水 | 屏東縣 | 佳樂水 | exact_title | 1.000 | true |
| 彰化孔子廟 | 彰化縣 | 彰化孔子廟 | exact_title | 1.000 | true |
| 三芝遊憩區 | 新北市 | 三芝區 | title_does_not_name_the_site | 0.000 | false |
| 新竹漁港 | 新竹市 | 新竹漁港 | exact_title | 1.000 | true |

配對欄位寫在 `data/v4_observational/phase18_external/page_matches.csv` 與 `pageviews_audit.json`：site_uid、page_id、article_title、language、match_evidence、earliest_available_month、monthly_coverage、missing_months、retrieval_timestamp、source_api、license。缺月清單只放在 JSON。

跨縣且覆蓋達標的條目有 4。它們分屬不同縣市，不是同一個選擇集合，也不能把原三個國家公園群補成 J=3。
