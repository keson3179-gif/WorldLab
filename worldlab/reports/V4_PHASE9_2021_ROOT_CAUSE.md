# V4 Phase 9｜2021 年年度合計根因

這一輪只新增衍生表，沒有改原始人次，也沒有把 version_difference 說成官方修訂。

重新計算後，月份合計與公布合計不一致的列數是 317。全部年份是 2021。月合計大於公布合計的有 317 筆。其他年份的不一致列數：{}。

根因次數：{'csv_published_equals_jan_to_oct_november_december_differ': 146, 'csv_published_equals_jan_to_oct_months_match_excel': 92, 'csv_published_equals_jan_to_oct_excel_unmatched': 79}。其中 unresolved 為真的有 225 筆。official_revision_claimed 全部為 false。

核心檢驗成立：這 317 筆的 CSV 公布合計都等於該列 1–10 月累計，不是十二個月合計。有些列因為中間月份是 0，較短的前綴也等於同一個數字，但 1–10 月累計一律相等。

92 筆優先組：CSV 與 Excel 十二個月逐月相同，Excel 自己的年度合計等於 Excel 的十二個月合計。差異只在 CSV 的合計欄被填成 1–10 月累計。這是欄位內容的算術結果，不是查到一份修訂公告。

146 筆：1–10 月與 Excel 逐月相同，11 月和 12 月不同，方向並不一致，也不是整表平移。CSV 合計欄仍然等於 1–10 月累計。沒有修訂紀錄，所以這 146 筆的月份差額維持 unresolved。

79 筆：同縣市同名對不到 2021 年 Excel。CSV 合計欄仍然等於 1–10 月累計。對不到的身分維持 unresolved。

每一筆的根因、證據與 unresolved 在 `V4_PHASE9_2021_ROOT_CAUSE_DETAIL.csv`。
