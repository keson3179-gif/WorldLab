# V4 Phase 15｜最小效用模型規格

三種層次都接在 Phase 14 的 observation adapter 上。agent_choice_probability、expected_model_presence_share、reported_visit_share 保持三個不同的量。一位旅客可以在同一個月出現在多個據點，所以模型份額對到官方回報份額只是對應假設，不是個人選擇機率。

M0 使用可觀察、而且屬於 historically_documented 的場所屬性。三群目前只有相同的統計類型符合這一類。沒有用常數填上視覺吸引或容量。三群的 M0 都是 prediction_unavailable。

M1 要加在 M0 之上，並使用截止月以前的回報人氣摘要。記憶規則沿用可適配的 LOGMAX、RW50、EXP99 介面，來源標成 reported_visit_share_conditional，exposure_claim 為 false。M0 沒有基準效用時，M1 也是 prediction_unavailable，沒有把歷史份額改名成 ABM 預測。

M2 只定義契約：中性場所代碼、凍結的場所卡版本、來源可追溯的描述、排除驗證期人氣與觀察期之後的描述、模型版本、提示雜湊、解析規則與回應保存。本輪 generation_status 是 not_run。九處裡只有同名配對的兩處有當期描述，而且那兩段文字的日期不是評估年。

沒有受訪者資料。合成偏好只在合成場所上檢查：屬性完全相同時，不同偏好權重得到相同份額。這不代表臺灣旅客人口。沒有獨立容量，所以不估計擁擠成本。同一條人氣序列不能同時當成吸引與擁擠。

| 地理群 | M0 | M0 原因 | M1 | M1 原因 |
| --- | --- | --- | --- | --- |
| 南投縣／玉山國家公園 | prediction_unavailable | no_differentiating_historically_documented_attribute | prediction_unavailable | base_utility_missing |
| 花蓮縣／太魯閣國家公園 | prediction_unavailable | no_differentiating_historically_documented_attribute | prediction_unavailable | base_utility_missing |
| 苗栗縣／雪霸國家公園 | prediction_unavailable | no_differentiating_historically_documented_attribute | prediction_unavailable | base_utility_missing |
