"""Read-only client for the PROD admin API.

Hard rules (see project memory `admin-api-prod-readonly-constraints`):
  - PROD ONLY. Stage is intentionally unsupported.
  - READ-ONLY. This module calls auth + read endpoints only. There is no
    write/update/delete method here, by design.
  - Credentials (username/password/2FA) are passed in per login from the UI —
    nothing is read from disk. The password is used once at login and never
    stored server-side; only the JWT is cached (to a gitignored file).

Login flow mirrors the admin front: pre-authenticate -> authenticate(2FA) ->
JWT. A cookie jar carries the JSESSIONID across the two calls so the 2FA step
sees the same session the browser would. After login the JWT goes in
`Authorization: Bearer` on every read call. CORS on prod is locked to the admin
domain, so these calls must be server-to-server (which is what this is).
"""

from __future__ import annotations

import base64
import http.cookiejar
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).parent
TOKEN_PATH = BASE_DIR / "storage" / ".admin_token.json"

# Prod only.
ADMIN_API_BASE = "https://adminv2.originals.games/admin/api"
ADMIN_ORIGIN = "https://adminv2.originals.games"
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Admin search "type" -> the statistic endpoint's path segment.
_STAT_PATH_TYPE = {"NICKNAME": "nickname", "UUID": "user", "EXTERNAL_ID": "external"}


class AdminError(Exception):
    """Raised on any admin-API failure. `need_login` is True when the failure
    is an auth problem (401/403) so the caller can prompt for a fresh 2FA."""

    def __init__(self, message: str, status: int | None = None, need_login: bool = False):
        super().__init__(message)
        self.status = status
        self.need_login = need_login


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    """Best-effort decode of a JWT payload segment (no signature check — we
    only want `exp` to know when to re-login)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:  # noqa: BLE001
        return {}


def _summarize_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Quick behavioral summary over the per-round history — enough to sanity
    check a pull and to seed persona generation later."""
    if not games:
        return {"rounds": 0}
    per_game: Counter = Counter()
    currencies: Counter = Counter()
    total_bet_eur = 0.0
    total_profit_eur = 0.0
    dates: list[str] = []
    for g in games:
        per_game[g.get("gameName") or "unknown"] += 1
        total_bet_eur += float(g.get("betAmountEur") or 0)
        total_profit_eur += float(g.get("profitEur") or 0)
        if g.get("betCurrency"):
            currencies[g["betCurrency"]] += 1
        if g.get("dateTime"):
            dates.append(g["dateTime"])
    return {
        "rounds": len(games),
        "distinct_games": len(per_game),
        "rounds_per_game": dict(per_game.most_common()),
        "currencies": dict(currencies),
        "total_bet_eur": round(total_bet_eur, 2),
        "total_profit_eur": round(total_profit_eur, 2),
        "avg_bet_eur": round(total_bet_eur / len(games), 4),
        "first_round": min(dates) if dates else None,
        "last_round": max(dates) if dates else None,
    }


class AdminClient:
    def __init__(self) -> None:
        self._token: str | None = None
        self._claims: dict[str, Any] = {}
        self._username: str | None = None
        self._casino: str | None = None
        # Shared cookie jar so pre-auth's JSESSIONID flows into authenticate.
        self._cookies = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookies)
        )
        self._load_token_cache()

    # ------------------------------------------------------------------ #
    # Token cache (gitignored)
    # ------------------------------------------------------------------ #
    def _load_token_cache(self) -> None:
        try:
            data = json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
            self._token = data.get("token")
            self._claims = data.get("claims", {})
            self._username = data.get("username")
            self._casino = data.get("casino")
        except (OSError, json.JSONDecodeError):
            pass

    def _save_token_cache(self) -> None:
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(
            json.dumps(
                {
                    "token": self._token,
                    "claims": self._claims,
                    "username": self._username,
                    "casino": self._casino,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #
    def _headers(self, auth: bool) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Origin": ADMIN_ORIGIN,
            "Referer": ADMIN_ORIGIN + "/",
            "User-Agent": _UA,
        }
        if auth and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _send(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: dict[str, Any] | None = None,
        auth: bool = True,
    ) -> dict[str, Any]:
        url = f"{ADMIN_API_BASE}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        headers = self._headers(auth=auth)
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with self._opener.open(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", "replace")
                resp_headers = {k: v for k, v in resp.headers.items()}
                status = resp.status
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            need = e.code in (401, 403)
            raise AdminError(
                f"admin API {e.code} on {method} /{path.lstrip('/')}: {raw[:200] or e.reason}",
                status=e.code,
                need_login=need,
            ) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise AdminError(f"admin API unreachable: {getattr(e, 'reason', e)}") from e

        parsed: Any = None
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
        return {"status": status, "json": parsed, "raw": raw, "headers": resp_headers}

    def _safe(self, fn: Callable[..., Any], *args: Any) -> tuple[Any, str | None]:
        """Run a read call; re-raise auth failures, but downgrade other errors
        to (None, message) so one failing section doesn't sink the whole pull."""
        try:
            return fn(*args), None
        except AdminError as e:
            if e.need_login:
                raise
            return None, str(e)

    # ------------------------------------------------------------------ #
    # Auth
    # ------------------------------------------------------------------ #
    def login(self, username: str, password: str, totp_code: str) -> dict[str, Any]:
        username = (username or "").strip()
        password = password or ""
        totp_code = (totp_code or "").strip()
        if not username or not password:
            raise AdminError("username and password required")
        if not totp_code:
            raise AdminError("2FA code required")

        # pre-authenticate: confirms the account + that 2FA is enrolled, and
        # seeds the session cookie the authenticate step is validated against.
        self._send("POST", "pre-authenticate", body={"username": username, "password": password}, auth=False)

        res = self._send(
            "POST",
            "authenticate",
            body={"username": username, "password": password, "twoFACode": totp_code},
            auth=False,
        )
        token = None
        body = res["json"] if isinstance(res["json"], dict) else {}
        token = body.get("token")
        if not token:
            hdr = res["headers"].get("X-Auth-Token") or res["headers"].get("Authorization")
            if hdr:
                token = hdr[7:] if hdr.lower().startswith("bearer ") else hdr
        if not token:
            raise AdminError(f"authenticate returned no token: {res['raw'][:200]}")

        self._token = token
        self._claims = _decode_jwt_claims(token)
        self._username = body.get("username") or username
        self._casino = body.get("casino")
        self._save_token_cache()
        return self.status()

    def status(self) -> dict[str, Any]:
        exp = self._claims.get("exp")
        now = int(time.time())
        return {
            "authenticated": bool(self._token) and (exp is None or exp > now),
            "username": self._username,
            "casino": self._casino,
            "token_expires_at": exp,
            "expires_in_seconds": (exp - now) if exp else None,
        }

    def _require_auth(self) -> None:
        if not self.status()["authenticated"]:
            raise AdminError("not authenticated (missing or expired token)", status=401, need_login=True)

    # ------------------------------------------------------------------ #
    # Read endpoints
    # ------------------------------------------------------------------ #
    def get_admin_details(self) -> dict[str, Any]:
        self._require_auth()
        return self._send("GET", "admin-details")["json"] or {}

    def resolve_casino_id(self, prefer: str | None = None) -> str | None:
        details = self.get_admin_details()
        casino_field = details.get("casino") or self._casino or ""
        ids = [c.strip() for c in casino_field.split(",") if c.strip()]
        if prefer and prefer in ids:
            return prefer
        return ids[0] if ids else None

    def get_casino_details(self, casino_id: str) -> dict[str, Any]:
        self._require_auth()
        return self._send("GET", f"casino/{casino_id}/details")["json"] or {}

    def search_users(self, casino_id: str, criteria: str, search_type: str = "NICKNAME") -> list[dict[str, Any]]:
        self._require_auth()
        res = self._send(
            "POST",
            f"search/casino/{casino_id}/users",
            body={"criteria": criteria, "type": search_type},
        )
        data = res["json"]
        return data if isinstance(data, list) else []

    def search_user_games(
        self,
        casino_id: str,
        criteria: str,
        date_from: str,
        date_to: str,
        search_type: str = "NICKNAME",
    ) -> list[dict[str, Any]]:
        self._require_auth()
        res = self._send(
            "POST",
            f"search/casino/{casino_id}/games",
            body={"criteria": criteria, "dateFrom": date_from, "dateTo": date_to, "type": search_type},
        )
        data = res["json"]
        if isinstance(data, dict):
            return data.get("userGames", [])
        return data if isinstance(data, list) else []

    def get_user_statistics(
        self,
        casino_id: str,
        criteria: str,
        search_type: str = "NICKNAME",
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        self._require_auth()
        path_type = _STAT_PATH_TYPE.get(search_type, "nickname")
        params: dict[str, Any] = {}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        path = f"statistic/casino/{casino_id}/{path_type}/{urllib.parse.quote(str(criteria))}"
        return self._send("GET", path, params=params or None)["json"] or {}

    def build_player_profile(
        self,
        criteria: str,
        date_from: str,
        date_to: str,
        search_type: str = "NICKNAME",
    ) -> dict[str, Any]:
        """One read sweep for a player: identity match + per-round history +
        aggregated statistics + a computed behavioral summary."""
        casino_id = self.resolve_casino_id()
        if not casino_id:
            raise AdminError("could not resolve casino id from admin-details")

        users, users_err = self._safe(self.search_users, casino_id, criteria, search_type)
        games, games_err = self._safe(
            self.search_user_games, casino_id, criteria, date_from, date_to, search_type
        )
        stats, stats_err = self._safe(
            self.get_user_statistics, casino_id, criteria, search_type, date_from, date_to
        )
        games = games or []

        profile: dict[str, Any] = {
            "casino_id": casino_id,
            "query": {
                "criteria": criteria,
                "type": search_type,
                "date_from": date_from,
                "date_to": date_to,
            },
            "matched_users": users or [],
            "game_count": len(games),
            "summary": _summarize_games(games),
            "statistics": stats or {},
            "games": games,
        }
        errors = {k: v for k, v in {"users": users_err, "games": games_err, "statistics": stats_err}.items() if v}
        if errors:
            profile["partial_errors"] = errors
        return profile
