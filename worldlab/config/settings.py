"""唯一可以寫模型名稱、temperature、seed 與 API endpoint 的地方。

第一階段不讀環境變數。一次 run 的參數以這個檔案加上程式版本為準。
修改 prompt 文字或 BehavioralProfile schema 時，請提高 prompt_version。
只改模型或抽樣參數時不必改 prompt_version；那些值會寫進 run metadata。
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_python() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("worldlab requires Python 3.11 or newer")


class Settings(BaseModel):
    """Phase-1 執行參數。欄位都有預設值，且實例建立後不可改。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    api_endpoint: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen3.5:9b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none"
    llm_seed: int = Field(
        default=42,
        ge=0,
        description="Sampling seed sent to the chat completion API. Does not draw traits.",
    )
    request_timeout_seconds: float = Field(default=180.0, gt=0.0)
    sdk_max_retries: int = Field(default=0, ge=0)

    master_seed: int = Field(
        default=42,
        ge=0,
        description="Seed for the Python trait RNG. The LLM does not use this to invent traits.",
    )
    prompt_version: str = "persona_behavioral_profile_v1"
    profile_language: str = "English"
    pilot_num_agents: int = Field(default=20, ge=1)

    scenario_seed: int = Field(default=42, ge=0)
    scenario_count: int = Field(default=30, ge=1)
    base_agent_count: int = Field(default=10, ge=1)
    crowding_min_gap: float = Field(default=0.35, gt=0.0, lt=1.0)
    crowd_aversion_low: float = Field(default=0.1, ge=0.0, le=1.0)
    crowd_aversion_mid: float = Field(default=0.5, ge=0.0, le=1.0)
    crowd_aversion_high: float = Field(default=0.9, ge=0.0, le=1.0)
    decision_max_attempts: int = Field(default=2, ge=1)
    decision_prompt_version: str = "place_choice_v1"
    calibration_prompt_version: str = "place_choice_v2"
    calibration_base_agent_count: int = Field(default=20, ge=1)

    world_seed: int = Field(
        default=42,
        ge=0,
        description="Seed for world placement and movement choice. Does not change the LLM sampling seed.",
    )
    choice_temperature: float = Field(
        default=100.0,
        gt=0.0,
        description="Softmax temperature on the 0-1000 overall_utility scale. Not the LLM temperature.",
    )
    world_timesteps: int = Field(default=100, ge=1)
    world_agent_count: int = Field(default=20, ge=1)

    personas_dir: Path = PROJECT_ROOT / "data" / "personas"
    logs_dir: Path = PROJECT_ROOT / "data" / "logs"

    @field_validator("api_endpoint")
    @classmethod
    def _openai_compatible_endpoint(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.endswith("/v1"):
            raise ValueError("api_endpoint must be an OpenAI-compatible base URL ending in /v1")
        return normalized

    @field_validator(
        "model",
        "api_key",
        "prompt_version",
        "profile_language",
        "decision_prompt_version",
        "calibration_prompt_version",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return self.__class__.model_validate(copied.model_dump())


def get_settings() -> Settings:
    ensure_python()
    return Settings()
