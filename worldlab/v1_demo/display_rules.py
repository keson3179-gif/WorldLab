"""Display rules for the V1 replay. These strings are the contract the page uses."""

from __future__ import annotations

NOT_APPLICABLE = "不適用／沒有合格樣本"
NO_EXIT_BY_HORIZON = "到 t=500 尚未跌破 18 人門檻"
NO_RELEASE_BY_HORIZON = "到 t=500 尚未解除集中"
EVENT_NOT_FORMED = "未形成"
LATE_HHI_LABEL = "第 101–500 步的累積訪問集中度"

TIME_MARKS = {
    "0": "t=0：初始位置",
    "50": "t=50：切換前最後一個 committed state",
    "51": "decision t=51 產生 committed state t=51。只有 EO 與 OE 會在這裡更換來源。",
}


def maintenance_text(maintenance: dict) -> dict[str, str]:
    if not maintenance.get("eligible"):
        return {
            "eligibility": NOT_APPLICABLE,
            "original_exit": NOT_APPLICABLE,
            "release_time": NOT_APPLICABLE,
        }
    exit_time = maintenance.get("original_exit")
    release_time = maintenance.get("release_time")
    return {
        "eligibility": "切換前符合維持分析資格",
        "original_exit": NO_EXIT_BY_HORIZON if exit_time is None else f"原場所退出於 committed t={exit_time}",
        "release_time": NO_RELEASE_BY_HORIZON if release_time is None else f"解除集中於 committed t={release_time}",
    }


def event_text(event: dict) -> dict[str, str]:
    if not event.get("occurred"):
        return {"start": EVENT_NOT_FORMED, "confirm": EVENT_NOT_FORMED, "place": EVENT_NOT_FORMED}
    return {
        "start": f"起始 committed t={event['start']}",
        "confirm": f"確認 committed t={event['confirm']}",
        "place": str(event.get("place")),
    }


def early_text(early_formed: bool) -> str:
    if early_formed:
        return "已在 committed t=50 以前確認"
    return "沒有在 committed t=50 以前確認"
