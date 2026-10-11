# V4 Phase 10｜來源欄位稽核

年度 Excel 沒有改。Phase 9 的 `v4_measurement_method_history.csv` 也沒有改。修正後的表是 `data/curated/v4_measurement_method_verified.csv`。

茂林國家風景區是來源裡真實的遊憩區標題列：縣市空白、月份空白，欄位 B 寫著這個名稱。它不是合併儲存格。合併的是欄 A 的類型，例如民國 100 年工作簿的 A156:A240 是「公立」，值只放在合併區的左上角。

舊解析器把最近一個標題一直往下帶。標題後面如果換了類型，而新類型的第一列本身就是有人次的據點、沒有新的遊憩區標題，茂林就會被填到花蓮、新北、臺中。這是程式填補，不是那些列的儲存格裡寫了茂林。

修正後仍標成茂林國家風景區的列數：120。其中縣市不是屏東或高雄的列數：0。無法在同一個類型區塊找到標題，或標題的風景區縣市對不上該列縣市時，parent_area 設為 unknown，列數 3484。沒有再用茂林或其他名稱填滿。文教場所區塊若在茂林標題後出現苗栗的臺灣客家文化館，該列與同一標題之後的列改為 unknown；標題原文留在 raw_parent_area。

據點名稱、縣市與遊客人次計算方式仍讀該列自己的儲存格。目前只能確認 parent_area 被錯填；不能因此說計數標籤或縣市也整欄錯誤。下面幾列是修正後的證據。

- 七星潭風景區／花蓮縣：來源列 169，parent_area=unknown，raw_parent_area=（空），計數標籤=資通訊設備，判定=unknown_no_header_in_category。
- 三峽老街／新北市：來源列 216，parent_area=unknown，raw_parent_area=（空），計數標籤=電信大數據，判定=unknown_no_header_in_category。
- 一中商圈／臺中市：來源列 309，parent_area=unknown，raw_parent_area=（空），計數標籤=電信大數據，判定=unknown_no_header_in_category。
- 茂林遊憩區／高雄市：來源列 112，parent_area=茂林國家風景區，raw_parent_area=茂林國家風景區，計數標籤=電信大數據，判定=same_category_header。
- 臺灣客家文化館／苗栗縣：來源列 172，parent_area=unknown，raw_parent_area=茂林國家風景區，計數標籤=電子計數器，判定=unknown_outside_header_counties。

每一列都有 source_file、source_sheet、source_row_number 與 raw_parent_area。raw_parent_area 只有在同一個類型區塊裡真的有標題時才填標題文字，否則留空。
