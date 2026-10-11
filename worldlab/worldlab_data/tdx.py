from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


TOKEN_URL = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
API_ROOT = "https://tdx.transportdata.tw/api/basic"
# Checked against the public basic-service Swagger (bike station example) on 2026-10-10.
# $top default 30; $skip is supported. Guest mode is documented as 20 calls per source IP per day.
# Member rate limits depend on the subscribed plan and are not a single fixed number.
DOCUMENTED_DATASETS = (
    {
        "dataset": "Bike/Station/City/{City}",
        "role": "public bike stations, current coordinates",
        "history": "static_current_proxy",
    },
    {
        "dataset": "historical bus timetable service",
        "role": "member historical files, example window cited publicly from 2021-01-21",
        "history": "not a reconstruction of 2018 or 2020 site accessibility",
    },
)


class TDXCredentialsError(RuntimeError):
    pass


class TDXRateLimitError(RuntimeError):
    pass


@dataclass
class TDXResponse:
    status: int
    body: bytes
    headers: dict[str, str]


class UrllibTransport:
    def post_form(self, url: str, form: dict[str, str]) -> TDXResponse:
        data = urllib.parse.urlencode(form).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        return self._send(request)

    def get(self, url: str, headers: dict[str, str]) -> TDXResponse:
        request = urllib.request.Request(url, method="GET")
        for key, value in headers.items():
            request.add_header(key, value)
        return self._send(request)

    def _send(self, request: urllib.request.Request) -> TDXResponse:
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return TDXResponse(response.status, response.read(), dict(response.headers))
        except urllib.error.HTTPError as exc:
            return TDXResponse(exc.code, exc.read(), dict(exc.headers))


class TDXClient:
    def __init__(self, client_id: str, client_secret: str, transport: UrllibTransport, cache_dir: Path, sleeper=time.sleep) -> None:
        if not client_id or not client_secret:
            raise TDXCredentialsError("TDX_CLIENT_ID 與 TDX_CLIENT_SECRET 尚未提供")
        self.client_id = client_id
        self.client_secret = client_secret
        self.transport = transport
        self.cache_dir = cache_dir
        self.sleeper = sleeper
        self.errors: list[str] = []
        self.token = ""

    def authenticate(self) -> None:
        response = self.transport.post_form(
            TOKEN_URL,
            {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        if response.status == 429:
            raise TDXRateLimitError("TDX token endpoint returned 429")
        if response.status != 200:
            raise TDXCredentialsError(f"TDX token status {response.status}")
        payload = json.loads(response.body.decode("utf-8"))
        self.token = str(payload.get("access_token", ""))
        if not self.token:
            raise TDXCredentialsError("TDX token response has no access_token")

    def get_json(self, path: str, query: dict[str, str] | None = None, retries: int = 3) -> dict[str, object]:
        url = path if path.startswith("http") else f"{API_ROOT}/{path.lstrip('/')}"
        if query:
            url = url + "?" + urllib.parse.urlencode(query)
        cache_path = self.cache_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json")
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if not self.token:
            self.authenticate()
        last_status = 0
        for attempt in range(retries):
            response = self.transport.get(url, {"Authorization": f"Bearer {self.token}"})
            last_status = response.status
            if response.status == 429:
                self.errors.append(f"429 {url} attempt {attempt + 1}")
                self.sleeper(min(2**attempt, 8))
                continue
            if response.status != 200:
                self.errors.append(f"{response.status} {url}")
                raise RuntimeError(f"TDX request failed with status {response.status}")
            payload = json.loads(response.body.decode("utf-8"))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload
        raise TDXRateLimitError(f"TDX rate limit persisted after retries, last status {last_status}")


def credentials_from_environ() -> tuple[str, str]:
    return os.environ.get("TDX_CLIENT_ID", "").strip(), os.environ.get("TDX_CLIENT_SECRET", "").strip()


def dry_run_plan() -> dict[str, object]:
    client_id, client_secret = credentials_from_environ()
    return {
        "will_call_network": False,
        "credentials_present": bool(client_id and client_secret),
        "token_url": TOKEN_URL,
        "api_root": API_ROOT,
        "datasets": list(DOCUMENTED_DATASETS),
        "pagination": {"top_parameter": "$top", "skip_parameter": "$skip", "documented_default_top": 30},
        "rate_limit": "訪客模式文件寫明每個來源 IP 每日至多 20 次；會員頻率依訂閱方案，不在此寫死。",
        "history_constraint": "目前端點查到的站點與即時可達性只標成 static_current_proxy，不能當成 2018 或 2020 的歷史交通條件。",
    }
