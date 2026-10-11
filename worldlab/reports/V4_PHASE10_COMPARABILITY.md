# V4 Phase 10｜分層可比性

A documented_comparable 需要文件把方法指到該據點該年。現行作業要點只列方法種類，沒有這種指認，所以相同標籤不會變成 A。

B reported_visits_conditional 是官方回報人次，可以做探索性份額，但必須帶來源與口徑限制，不能進入正式確認性預測。

C not_comparable 含停車、住宿、電信等代理，或代理與直接計數混在同一群。這些群只能看個別序列。

地理群與替代群的等級次數：{'C': 35, 'B': 9}。替代群即使測到 B，也因為替代關係未確認而不做份額。

探索性份額列數：2185。檔案若存在，dataset_role 是 conditional_exploratory。

| 群組 | 定義 | 等級 | 共同月份 | 衝突比例 | 識別信心 | 探索性份額 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 南投縣／日月潭國家風景區 | A_geographic | C | 96 | 0.0101 | 1.00 | False |
| 南投縣／玉山國家公園 | A_geographic | B | 120 | 0.0000 | 1.00 | True |
| 嘉義縣／阿里山國家風景區 | A_geographic | B | 46 | 0.0143 | 1.00 | True |
| 基隆市／北海岸及觀音山國家風景區 | A_geographic | C | 144 | 0.0000 | 1.00 | False |
| 宜蘭縣／東北角暨宜蘭海岸國家風景區 | A_geographic | C | 0 | 0.0000 | 1.00 | False |
| 屏東縣／墾丁國家公園 | A_geographic | C | 144 | 0.0000 | 1.00 | False |
| 屏東縣／大鵬灣國家風景區 | A_geographic | C | 154 | 0.0128 | 1.00 | False |
| 屏東縣／茂林國家風景區 | A_geographic | C | 0 | 0.0154 | 1.00 | False |
| 新北市／北海岸及觀音山國家風景區 | A_geographic | C | 144 | 0.0000 | 1.00 | False |
| 新北市／東北角暨宜蘭海岸國家風景區 | A_geographic | C | 0 | 0.0000 | 1.00 | False |
| 澎湖縣／澎湖國家風景區 | A_geographic | C | 22 | 0.0078 | 1.00 | False |
| 臺中市／參山國家風景區 | A_geographic | C | 142 | 0.0139 | 1.00 | False |
| 臺北市／陽明山國家公園 | A_geographic | C | 72 | 0.0000 | 1.00 | False |
| 臺南市／西拉雅國家風景區 | A_geographic | C | 142 | 0.0139 | 1.00 | False |
| 臺南市／雲嘉南濱海國家風景區 | A_geographic | C | 94 | 0.0079 | 1.00 | False |
| 臺東縣／東部海岸國家風景區 | A_geographic | C | 120 | 0.0000 | 1.00 | False |
| 臺東縣／花東縱谷國家風景區 | A_geographic | C | 22 | 0.0167 | 1.00 | False |
| 花蓮縣／太魯閣國家公園 | A_geographic | B | 144 | 0.0000 | 1.00 | True |
| 花蓮縣／東部海岸國家風景區 | A_geographic | C | 120 | 0.0000 | 1.00 | False |
| 花蓮縣／花東縱谷國家風景區 | A_geographic | C | 22 | 0.0119 | 0.75 | False |
| 苗栗縣／雪霸國家公園 | A_geographic | B | 144 | 0.0000 | 1.00 | True |
| 連江縣／馬祖國家風景區 | A_geographic | C | 144 | 0.0000 | 1.00 | False |
| 金門縣／金門國家公園 | A_geographic | B | 72 | 0.0000 | 1.00 | True |
| 高雄市／茂林國家風景區 | A_geographic | C | 142 | 0.0139 | 1.00 | False |
| 南投縣／日月潭國家風景區 | B_substitution_candidate | C | 96 | 0.0101 | 1.00 | False |
| 南投縣／玉山國家公園 | B_substitution_candidate | B | 120 | 0.0000 | 1.00 | False |
| 嘉義縣／阿里山國家風景區 | B_substitution_candidate | B | 46 | 0.0119 | 1.00 | False |
| 基隆市／北海岸及觀音山國家風景區 | B_substitution_candidate | C | 144 | 0.0000 | 1.00 | False |
| 宜蘭縣／東北角暨宜蘭海岸國家風景區 | B_substitution_candidate | C | 0 | 0.0000 | 1.00 | False |
| 屏東縣／墾丁國家公園 | B_substitution_candidate | C | 144 | 0.0000 | 1.00 | False |
| 屏東縣／茂林國家風景區 | B_substitution_candidate | C | 0 | 0.0154 | 1.00 | False |
| 新北市／北海岸及觀音山國家風景區 | B_substitution_candidate | C | 144 | 0.0000 | 1.00 | False |
| 新北市／東北角暨宜蘭海岸國家風景區 | B_substitution_candidate | C | 0 | 0.0000 | 1.00 | False |
| 澎湖縣／澎湖國家風景區 | B_substitution_candidate | C | 22 | 0.0078 | 1.00 | False |
| 臺中市／參山國家風景區 | B_substitution_candidate | C | 142 | 0.0139 | 1.00 | False |
| 臺北市／陽明山國家公園 | B_substitution_candidate | C | 72 | 0.0000 | 1.00 | False |
| 臺南市／西拉雅國家風景區 | B_substitution_candidate | C | 142 | 0.0139 | 1.00 | False |
| 臺南市／雲嘉南濱海國家風景區 | B_substitution_candidate | C | 94 | 0.0104 | 1.00 | False |
| 臺東縣／東部海岸國家風景區 | B_substitution_candidate | C | 120 | 0.0000 | 1.00 | False |
| 臺東縣／花東縱谷國家風景區 | B_substitution_candidate | C | 22 | 0.0167 | 1.00 | False |
| 花蓮縣／太魯閣國家公園 | B_substitution_candidate | B | 144 | 0.0000 | 1.00 | False |
| 花蓮縣／花東縱谷國家風景區 | B_substitution_candidate | C | 22 | 0.0119 | 0.75 | False |
| 金門縣／金門國家公園 | B_substitution_candidate | B | 72 | 0.0000 | 1.00 | False |
| 高雄市／茂林國家風景區 | B_substitution_candidate | C | 142 | 0.0139 | 1.00 | False |
