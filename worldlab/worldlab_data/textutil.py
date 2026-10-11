from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


READ_ENCODINGS = ("utf-8-sig", "utf-8", "cp950")
MISSING_TOKENS = {"", "-", "—", "–", "…", "...", "na", "n/a", "null", "none", "nan", ".", "*"}
CLOSED_TOKENS = {"停業", "休園", "停止營業", "暫停開放", "closed"}
NOT_IN_STATS_TOKENS = {"未統計", "未納入", "不統計", "尚未納入"}


def file_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_text_with_encoding(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    last_error: Exception | None = None
    for encoding in READ_ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "unknown",
        raw[:40],
        0,
        min(len(raw), 40),
        f"unable to decode {path.name}: {last_error}",
    )


def detect_encoding(path: Path) -> str:
    _, encoding = read_text_with_encoding(path)
    return encoding


def nfkc(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def county_parts(value: object) -> tuple[str, str]:
    """Split a bilingual cell. The Chinese line stays the county; English is kept aside."""
    text = nfkc(value)
    if not text:
        return "", ""
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    chinese = lines[0] if lines else ""
    english = lines[1] if len(lines) > 1 else ""
    chinese = chinese.replace("台", "臺")
    return chinese, english


def comparison_key(value: object) -> str:
    text = nfkc(value).replace("台", "臺")
    text = re.sub(r"[\s·・‧,，.。、/／\\\-—–()（）\[\]【】'\"「」]", "", text)
    return text.casefold()


def parse_calendar_year(value: object) -> int | None:
    """Accept a western year, or an ROC year from 1 to 200."""
    text = nfkc(value)
    if not text:
        return None
    matched = re.search(r"-?\d+", text)
    if matched is None:
        return None
    year = int(matched.group(0))
    if 1 <= year <= 200:
        return year + 1911
    if year >= 1912:
        return year
    return None


def parse_count(value: object) -> tuple[float | None, str]:
    """Return a visit count and an observation status. Missing values stay missing."""
    if value is None:
        return None, "missing"
    text = nfkc(value).replace(",", "")
    if text.casefold() in MISSING_TOKENS:
        return None, "missing" if text == "" else "missing_token"
    compact = text.casefold().replace(" ", "")
    if compact in CLOSED_TOKENS:
        return None, "closed_reported"
    if compact in NOT_IN_STATS_TOKENS:
        return None, "not_in_statistics"
    try:
        number = float(text)
    except ValueError:
        return None, "unparsed_token"
    if number < 0:
        return None, "invalid_negative"
    if number == 0:
        return 0.0, "reported_zero"
    return number, "reported"


def stable_site_id(parts: list[str]) -> str:
    payload = "\x1f".join(comparison_key(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"stat_{digest}"
