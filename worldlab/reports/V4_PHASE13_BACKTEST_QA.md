# V4 Phase 13｜回測 QA

這一輪讀取的是 Phase 12 的探索性候選面板，沒有讀取 2025–2026 保留檔的人次、份額或 HHI。阿里山列留在原面板，沒有進入本輪預測。

Python 3.12.10，平台 Windows-11-10.0.26200-SP0，pandas 3.0.6，pyarrow 26.0.0。

預測列 16730。已預測的據點列 16730。prediction_unavailable 列 0。
份額向量加總檢查失敗的群月數 0。截止日或來源月越界的列數 0。2025 以後的預測月 沒有。

程式內對每個預測另用截止日以前的歷史重算一次。兩次份額或來源月不一致會中止寫出結果。本輪寫出前沒有觸發這項中止。

共同評分月是同一情境、同一群裡七個基準都成功預測的月份。比較表使用這組月份。

| 地理群 | 情境 | 可評估月 | 共同評分月 | 描述月份數 | 滾動起點 | 非重疊區塊 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 南投縣／玉山國家公園 | ex_post_max_information | 96 | 96 | 96 | 85 | 8 |
| 南投縣／玉山國家公園 | assumed_publication_delay | 96 | 96 | 96 | 85 | 8 |
| 花蓮縣／太魯閣國家公園 | ex_post_max_information | 120 | 120 | 120 | 109 | 10 |
| 花蓮縣／太魯閣國家公園 | assumed_publication_delay | 120 | 120 | 120 | 109 | 10 |
| 苗栗縣／雪霸國家公園 | ex_post_max_information | 120 | 120 | 120 | 109 | 10 |
| 苗栗縣／雪霸國家公園 | assumed_publication_delay | 120 | 120 | 120 | 109 | 10 |
| 金門縣／金門國家公園 | ex_post_max_information | 17 | 17 | 17 | 6 | 1 |
| 金門縣／金門國家公園 | assumed_publication_delay | 17 | 17 | 17 | 6 | 1 |

描述月份數等於不重複日曆月。它不是滾動起點數，也不是起點數乘上 12。

新增測試：test_share_prediction_simplex、test_rolling_origin_no_leakage、test_asof_lag_one、test_asof_lag_three、test_missing_month_no_imputation、test_all_zero_month_exclusion、test_fixed_group_membership、test_same_scoring_calendar、test_no_2025_data_access、test_overlapping_windows_not_independent、test_2024_identity_evidence_separation。

輸入 SHA-256：

- `data/curated/v4_forecast_panel_candidate.parquet`：`18b35ccc7bc0c2fe60c01e0a6bc0b5258069db0c2f7acdf8bbd89199934866d3`
- `data/curated/v4_2024_identity_audit.csv`：`7fb9d4af71be7ea8330a316e05aa535b0bbeb3f5db348ccfc43e057d6a0daec3`
- `data/curated/v4_2021_alignment_audit.csv`：`d1265909843261ab28819c5e371d702ab654cf3d8abfd034a18db0cd1c4f5eab`
- `data/curated/v4_exploratory_visit_shares_conditional.csv`：`0f28543f8690561a8e97229e05516eb948bbe30436c0118dcb08fecfe4ef5aa8`
- `data/curated/v4_month_validity.csv`：`d8312862a7446451c4830249c58d48b696a8bedca231bffa93545a7f766ba24f`
- `reports/V4_PHASE12_BASELINE_PROTOCOL.md`：`9ef0f4f1c18760cdafbd3ac0d05fd161b128a3e3206741a14b17544377e4045f`
- `reports/V4_PHASE12_IDENTITY_READINESS.md`：`ac5412e955e2ca2aee0f547af7bf5a1dfd47052c10c577a1116845a628a44df8`
- `reports/V4_PHASE12_CATEGORY_ROOT_CAUSE.md`：`496216a4940f3daf39cccf6a0a6f15cf4e69e5a4ef30b5dd5fefd2d5e94cae21`
- `reports/V4_PHASE12_2021_ALIGNMENT.md`：`be6b7625d3a2bb01ef8467ea633a158b038dc9659574478f14b618ab6fde9d49`
- `reports/V4_PHASE11_FORECAST_FEASIBILITY.md`：`96b5afb7463d2a3b8ef1a8ef7c518a11819462ddb6d23718f585ab26c39816fd`
- `reports/V4_PHASE11_CONCENTRATION_EXPLORATION.md`：`7945923297c476ee85546a778a8d00bede844bda4a0e4a1d9c94dc6234cf2afb`

圖：
- `figures/v4_baselines/share_tv_scenario_a.png`
- `figures/v4_baselines/share_mae_scenario_a.png`
- `figures/v4_baselines/share_tv_delay_increase.png`

評估程式 SHA-256：phase13.py `b7d076d8994a1905ef7ae85ea8e6c52ee45e89882194274419acfb15fbd19244`，phase13_run.py `199e4d7fe49490b10c6b7236d58c8e15991813aa5491270c70f6a9783825af6f`。

受保護的 Phase 11–12 檔案在寫出 Phase 13 前後雜湊相同。
