# V4 Phase 12｜2021 年 11–12 月對齊

阿里山這幾列的 Excel 人次，就在該據點那一列的 110年11月、110年12月儲存格。月份欄沒有合併。阿里山標題列在第 97 列，縣市和月份是空的，它是遊憩區標題，不是插在據點之間的空白列。

解析值等於該列儲存格，所以這不是解析器把上一列讀成這一列。CSV 與 Excel 是兩套發布值。兩套都保留，chosen_version 留空。

相鄰列的數字可以相同。這個相同只描述模式，不能單獨證明正確配對，也沒有用來改寫任何月份。

B 類 146 筆的模式次數：{'different': 102, 'equals_previous_visit_row': 42, 'equals_next_visit_row': 2}。parser_error 筆數是 0。

## 阿里山

- 觸口遊客中心：Excel 列 100，CSV 11月 11939.0、Excel 11月 4175.0，上一筆 達娜伊谷，模式 equals_previous_visit_row。
- 梅山太平雲梯：Excel 列 101，CSV 11月 4175.0、Excel 11月 53180.0，上一筆 觸口遊客中心，模式 equals_previous_visit_row。
- 達娜伊谷：Excel 列 99，CSV 11月 8728.0、Excel 11月 11939.0，上一筆 圓潭生態園區，模式 equals_previous_visit_row。

圓潭自然生態園區的 CSV 名稱對不到 Excel 的「圓潭生態園區」，所以維持 excel_name_unmatched。沒有因為數字接近就配對。

同一張工作簿裡，其他 B 類據點若也出現「CSV 兩個月都等於上一筆 Excel」的模式，次數在上面的 pattern 統計。那仍是來源之間的對照，不是已確認的解析錯誤。
