# V4 Phase 14｜觀察轉接架構

這一輪是 exploratory development。沒有執行 P4，沒有呼叫 LLM，也沒有讀取 2025–2026 保留人次。

原始核心與 Experiment 001–020 沒有改寫。舊 Atlas 的 site_01 到 site_05 是五個合成場所，不能改名成玉山、太魯閣或雪霸的效用。

## 可直接重用的介面

- `world.choice.softmax_choice`
- `world.choice.choice_seed`
- `world.choice.agent_reconsiders`
- `world.experiment_002.REFERENCE_CHOICE_TEMPERATURE`
- `world.experiment_002.REFERENCE_RECONSIDERATION_RATE`
- `world.experiment_004.popularity_scores`
- `world.experiment_004.SocialState`
- `world.experiment_004.commit_social`
- `world.experiment_005.centered_scores`
- `world.experiment_005.RollingMemory`
- `world.experiment_005.DecayMemory`
- `world.experiment_017.score_memory`
- `world.experiment_018.q_components`
- `world.experiment_020.source_at`
- `world.geography.PLACE_IDS`
- `worldlab_data.phase13.predict_shares`
- `worldlab_data.phase13.information_cutoff`

這些函式仍服務 J=5 的長期模型。觀察層只讀它們的契約，不把呼叫結果寫回 data/worlds。

## 資料流

WorldLab simulation state → agent probability or presence → observation aggregation → monthly predicted share → comparison with reported_visit_distribution。

模式 A conditional_one_step 用截止月以前的真實回報份額建立狀態，預測下一個月，預測值不寫回觀測歷史。模式 B free_running 只接受 model_endogenous_presence。本輪模式 B 只在合成資料上做 smoke test。

社會訊號必須帶來源標籤。回報份額的標籤是 reported_visit_share_conditional，exposure_claim 固定為 false。

選擇溫度、再考慮率與分數截斷沿用引擎常數：{'choice_temperature': 200.0, 'reconsideration_rate': 0.5, 'score_clip': 2.0, 'agent_count_in_engine': 20, 'baseline_models': ['persistence', 'same_month_last_year', 'seasonal_month_mean', 'trailing_3_mean', 'trailing_12_mean', 'fixed_equal_weight', 'uniform_reference'], 'required_replicates_for_component_se_0_01': 125}。
