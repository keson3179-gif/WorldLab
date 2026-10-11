# V4 Phase 17｜外部特徵盤點

固定群組是玉山、太魯閣、雪霸，各 J=3。沒有依 Phase 16 的誤差改群。特徵矩陣沒有讀取回報份額，也沒有填補缺值。

週末天數來自曆法，辦公日曆在 2024 年以前用到的檔案有 7 個。沒有本地檔的年份是 [2012, 2013, 2014, 2015, 2016]。星期比對不一致 0 天。

v0.2 氣象主站對上的據點是：太魯閣國家公園遊客中心、雪見遊憩區。沒有對上的是：塔塔加遊憩區、玉山管理處遊客服務中心、排雲山莊、臺八線沿線景觀區、布洛灣遊憩區、觀霧遊客中心、汶水遊客中心。景點高程沒有檢查。氣候距平欄沒有使用。候選矩陣只保留 2024 年以前的測站月。

活動清單更新時間 2026-10-10T14:30:17+08:00，共 1137 筆，開始月 2025-01 至 2026-12。開發月份裡沒有可回接的同名活動。

TDX 狀態 offline_credentials_absent。網路關注資料狀態 external_social_information_unobserved。

| 特徵 | 地理群 | 有值據點 | 月份 | 覆蓋率 | 群內變異 | 時間變異 | 嚴格截止 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weekend_days | 玉山國家公園 | 3 | 120 | 1.000 | false | true | true |
| weekend_days | 太魯閣國家公園 | 3 | 144 | 1.000 | false | true | true |
| weekend_days | 雪霸國家公園 | 3 | 144 | 1.000 | false | true | true |
| named_holiday_days | 玉山國家公園 | 3 | 60 | 0.500 | false | true | false |
| named_holiday_days | 太魯閣國家公園 | 3 | 84 | 0.583 | false | true | false |
| named_holiday_days | 雪霸國家公園 | 3 | 84 | 0.583 | false | true | false |
| same_month_mean_daily_max_c | 玉山國家公園 | 0 | 0 | 0.000 | false | false | false |
| same_month_mean_daily_max_c | 太魯閣國家公園 | 1 | 103 | 0.238 | false | true | false |
| same_month_mean_daily_max_c | 雪霸國家公園 | 1 | 108 | 0.250 | false | true | false |
| lagged_mean_daily_max_c | 玉山國家公園 | 0 | 0 | 0.000 | false | false | false |
| lagged_mean_daily_max_c | 太魯閣國家公園 | 1 | 102 | 0.236 | false | true | false |
| lagged_mean_daily_max_c | 雪霸國家公園 | 1 | 107 | 0.248 | false | true | false |
