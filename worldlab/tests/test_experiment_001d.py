from __future__ import annotations

from agents.persona_generator import generate_traits
from config.settings import get_settings
from experiments.decision_engine import build_decision_messages
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001b import OUTPUT_DIR as EXPERIMENT_001B_DIR
from experiments.experiment_001b import calibration_trait_definitions
from experiments.experiment_001c import OUTPUT_DIR as EXPERIMENT_001C_DIR
from experiments.experiment_001d import (
    EXPECTED_DECISIONS,
    OPTION_1,
    OPTION_2,
    OUTPUT_DIR,
    PlaceChoiceDecisionEnumBA,
    build_design,
    build_factorial_messages,
    fit_decomposition_models,
    select_scenarios,
    substantive_choice,
    summarize_decomposition,
    _tasks,
)
from experiments.tradeoff_scenarios import generate_scenario_matrix
from models.schemas import PlaceChoiceDecision, export_response_schema


def test_sample_covers_blocks_and_cost_levels_without_a_new_draw() -> None:
    matrix = generate_scenario_matrix()
    chosen = select_scenarios(matrix)
    assert len(chosen) == 12
    assert {(row["tradeoff_block"], row["avoidance_cost_level"]) for row in chosen} == {
        (block, cost) for block in ("A", "B", "C", "D") for cost in (1, 4, 6)
    }
    by_id = {row["scenario_id"]: row for row in matrix["scenarios"]}
    for row in chosen:
        assert row == by_id[row["scenario_id"]]
    design = build_design(chosen)
    assert design["design_sha256"] == build_design(chosen)["design_sha256"]
    assert design["schema_choice_enum"] == {"AB": ["A", "B"], "BA": ["B", "A"]}
    assert len(design["factorial_cells"]) == 8


def test_schema_enum_order_flips_only_the_choice_field() -> None:
    original = export_response_schema(PlaceChoiceDecision)
    flipped = export_response_schema(PlaceChoiceDecisionEnumBA)
    assert original["properties"]["choice"]["enum"] == ["A", "B"]
    assert flipped["properties"]["choice"]["enum"] == ["B", "A"]
    assert original["properties"]["reason_codes"] == flipped["properties"]["reason_codes"]


def test_prompt_changes_only_option_order_and_label() -> None:
    traits, _ = generate_traits(42, 0)
    pair = build_design(select_scenarios(generate_scenario_matrix()))["pairs"][0]
    base = build_decision_messages(
        traits,
        pair["physical_option_1"],
        pair["physical_option_2"],
        decision_prompt_version="place_choice_v2",
        trait_definitions=calibration_trait_definitions(),
    )
    labeled_b_second = build_factorial_messages(
        traits,
        pair,
        position="second",
        label="B",
        decision_prompt_version="place_choice_v2",
    )
    assert labeled_b_second[0]["content"] == base[0]["content"]
    text = labeled_b_second[1]["content"]
    assert text.index("OPTION A") < text.index("OPTION B")
    assert "PLACE_A" not in text
    assert "Choose A or B." in text
    assert "almost no negative effect" in text
    for banned in (
        "ignore ordering",
        "avoid bias",
        "carefully compare",
        "do not prefer",
        "first option",
        "physical_option",
        "hypothesis",
        "tradeoff",
        "avoidance_cost",
    ):
        assert banned not in text
    again = build_factorial_messages(
        traits,
        pair,
        position="second",
        label="B",
        decision_prompt_version="place_choice_v2",
    )
    assert again == labeled_b_second
    option_1 = _block(text, "B")
    option_2 = _block(text, "A")
    assert f'"crowding": {pair["physical_option_1"]["crowding"]}' in option_1
    assert f'"crowding": {pair["physical_option_2"]["crowding"]}' in option_2


def test_substantive_choice_follows_the_assigned_label() -> None:
    assert substantive_choice("B", "B") == OPTION_1
    assert substantive_choice("A", "B") == OPTION_2
    assert substantive_choice("A", "A") == OPTION_1


def test_grid_size_and_output_directories() -> None:
    settings = get_settings()
    assert settings.temperature == 0.0
    assert settings.llm_seed == 42
    assert EXPECTED_DECISIONS == 2880
    assert len({OUTPUT_DIR, EXPERIMENT_001_DIR, EXPERIMENT_001B_DIR, EXPERIMENT_001C_DIR}) == 4
    bases = [{"base_agent_id": f"agent_{index:04d}", "traits": generate_traits(42, index)[0]} for index in range(10)]
    pairs = build_design(select_scenarios(generate_scenario_matrix()))["pairs"]
    assert len(_tasks(bases, pairs)) == 2880


def test_diagnostics_separate_position_from_label_and_enum() -> None:
    rows = _always_first_rows()
    summary = summarize_decomposition(rows, matrix_sha256="abc", design_sha256="def", expected_decision_count=len(rows))
    assert summary["first_position_advantage"]["difference"] == 1
    assert summary["a_label_advantage"]["difference"] == 0
    assert summary["enum_order_advantage"]["difference"] == 0
    assert summary["substantive_invariance"]["rate"] == 0
    assert summary["substantive_invariance"]["n"] == 2 * 3 * 12 * 4
    assert summary["pass_threshold"] is None


def test_models_report_predicted_differences() -> None:
    rows = _threshold_rows()
    regression = fit_decomposition_models(rows)
    main = regression["main_model"]
    interaction = regression["interaction_model"]
    assert regression["analysis_level"] == "pilot"
    assert main["status"] == "ok"
    assert interaction["status"] == "ok"
    first = main["technical_effects"]["physical_option_1_is_first"]
    label = main["technical_effects"]["physical_option_1_is_A"]
    enum_effect = main["technical_effects"]["schema_A_first"]
    assert first["predicted_difference"] > 0
    assert label["predicted_difference"] > 0
    assert enum_effect["predicted_difference"] > 0
    assert first["coefficient"]["odds_ratio"] > 1
    assert set(main["crowd_aversion_predictions"]) == {"0.00", "0.50", "1.00"}
    assert main["crowd_aversion_predictions"]["1.00"] > main["crowd_aversion_predictions"]["0.00"]
    for name in (
        "physical_option_1_is_first:physical_option_1_is_A",
        "physical_option_1_is_first:schema_A_first",
        "physical_option_1_is_A:schema_A_first",
    ):
        assert name in interaction["interaction_effects"]
        assert "coefficient" in interaction["interaction_effects"][name]
    assert set(interaction["interaction_probability_contrasts"]) == {
        "physical_option_1_is_first:physical_option_1_is_A",
        "physical_option_1_is_first:schema_A_first",
        "physical_option_1_is_A:schema_A_first",
    }


def _always_first_rows() -> list[dict[str, object]]:
    rows = []
    for row in _grid():
        chooses_option_1 = row["physical_option_1_position"] == "first"
        label = row["physical_option_1_label"]
        raw = label if chooses_option_1 else ("B" if label == "A" else "A")
        row["raw_choice"] = raw
        row["substantive_choice"] = OPTION_1 if chooses_option_1 else OPTION_2
        row["chosen_physical_option_1"] = int(chooses_option_1)
        row["chosen_lower_crowding"] = int(row["substantive_choice"] == row["lower_crowding_substantive_id"])
        row["schema_valid"] = True
        row["schema_failure_count"] = 0
        row["retry_count"] = 0
        rows.append(row)
    return rows


def _threshold_rows() -> list[dict[str, object]]:
    rows = []
    for row in _grid(agents=2):
        is_first = int(row["physical_option_1_position"] == "first")
        is_a = int(row["physical_option_1_label"] == "A")
        schema_a = int(row["schema_enum_order"] == ["A", "B"])
        level_high = int(row["crowd_aversion"] == 1.0)
        chosen = int(is_first + is_a + schema_a + level_high >= 2)
        label = row["physical_option_1_label"]
        raw = label if chosen else ("B" if label == "A" else "A")
        row["physical_option_1_is_first"] = is_first
        row["physical_option_1_is_A"] = is_a
        row["schema_A_first"] = schema_a
        row["raw_choice"] = raw
        row["substantive_choice"] = OPTION_1 if chosen else OPTION_2
        row["chosen_physical_option_1"] = chosen
        row["chosen_lower_crowding"] = int(row["substantive_choice"] == row["lower_crowding_substantive_id"])
        row["schema_valid"] = True
        row["schema_failure_count"] = 0
        row["retry_count"] = 0
        rows.append(row)
    return rows


def _grid(agents: int = 2) -> list[dict[str, object]]:
    pairs = build_design(select_scenarios(generate_scenario_matrix()))["pairs"]
    rows = []
    for agent_index in range(agents):
        for level in (0.0, 0.5, 1.0):
            for pair in pairs:
                for position in ("first", "second"):
                    for label in ("A", "B"):
                        for enum_order in (["A", "B"], ["B", "A"]):
                            rows.append(
                                {
                                    "base_agent_id": f"agent_{agent_index:04d}",
                                    "crowd_aversion": level,
                                    "scenario_id": pair["scenario_id"],
                                    "tradeoff_block": pair["tradeoff_block"],
                                    "avoidance_cost_level": pair["avoidance_cost_level"],
                                    "lower_crowding_substantive_id": pair["lower_crowding_substantive_id"],
                                    "physical_option_1_position": position,
                                    "physical_option_1_label": label,
                                    "schema_enum_order": enum_order,
                                    "physical_option_1_is_first": int(position == "first"),
                                    "physical_option_1_is_A": int(label == "A"),
                                    "schema_A_first": int(enum_order == ["A", "B"]),
                                }
                            )
    return rows


def _block(text: str, label: str) -> str:
    start = text.index(f"OPTION {label}\n")
    end = text.index(f"END_OPTION_{label}")
    return text[start:end]
