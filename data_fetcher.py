"""
Mundel データフェッチャー: 経済データの取得・キャッシュ

FRED API と yfinance を使用してマクロ指標と為替レートを取得し、
SQLite で1時間キャッシュする。
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests
import yfinance as yf
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 環境変数の読み込み
# -----------------------------------------------------------------------------
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# data/ ディレクトリと SQLite DB のパス
DATA_DIR = Path("/tmp/mundel_data")
DB_PATH = DATA_DIR / "mundel_cache.db"
CACHE_TTL_SECONDS = 3600  # 1時間

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# マクロ指標の FRED シリーズID定義
# -----------------------------------------------------------------------------
MACRO_SERIES = {
    "FEDFUNDS": "米国政策金利",
    "CPIAUCSL": "米国消費者物価指数(CPI)",
    "IRSTCB01JPM156N": "日本政策金利",
    "JPNCPIALLMINMEI": "日本消費者物価指数(CPI)",
}


def _ensure_data_dir() -> None:
    """data/ ディレクトリが存在することを保証する"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_db_connection() -> sqlite3.Connection:
    """SQLite 接続を取得し、キャッシュテーブルを初期化する"""
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            data_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _get_cached(cache_key: str) -> Optional[dict]:
    """
    キャッシュからデータを取得する。
    1時間以内のデータがあれば返す。なければ None。
    """
    try:
        conn = _get_db_connection()
        cursor = conn.execute(
            "SELECT data_json, fetched_at FROM cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        data_json, fetched_at_str = row
        fetched_at = datetime.fromisoformat(fetched_at_str)
        if datetime.now() - fetched_at < timedelta(seconds=CACHE_TTL_SECONDS):
            return json.loads(data_json)
        return None
    except Exception as e:
        logger.warning("キャッシュ読み込みに失敗しました: %s", e)
        return None


def _set_cached(cache_key: str, data: dict) -> None:
    """キャッシュにデータを保存する"""
    try:
        conn = _get_db_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO cache (cache_key, data_json, fetched_at)
            VALUES (?, ?, ?)
            """,
            (cache_key, json.dumps(data, ensure_ascii=False), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("キャッシュ保存に失敗しました: %s", e)


def get_exchange_rate(pair: str = "USDJPY=X") -> dict[str, Any]:
    """
    yfinance から為替レートを取得する。

    Args:
        pair: 通貨ペアのティッカー（デフォルト: USDJPY=X）

    Returns:
        {
            "pair": str,
            "current_price": float | None,
            "closes_7d": list[{"date": str, "close": float}],
            "fetched_at": str,
            "error": str | None  # エラー時のみ
        }
    """
    cache_key = f"exchange:{pair}"
    cached = _get_cached(cache_key)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    result: dict[str, Any] = {
        "pair": pair,
        "current_price": None,
        "closes_7d": [],
        "fetched_at": datetime.now().isoformat(),
        "from_cache": False,
        "error": None,
    }

    try:
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="7d")

        if hist.empty:
            result["error"] = "為替データが取得できませんでした"
            return result

        # 終値のリスト（日付昇順）
        closes_7d = [
            {"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"])}
            for idx, row in hist.iterrows()
        ]
        result["closes_7d"] = closes_7d

        # 現在価格（直近の終値、または regularMarketPrice）
        if closes_7d:
            result["current_price"] = closes_7d[-1]["close"]
        else:
            info = ticker.info
            result["current_price"] = info.get("regularMarketPrice") or info.get("previousClose")

        if result["error"] is None:
            _set_cached(cache_key, result)

    except Exception as e:
        logger.exception("為替データ取得中にエラーが発生しました")
        result["error"] = str(e)

    return result


def get_macro_indicators() -> dict[str, Any]:
    """
    FRED API からマクロ指標を取得する。

    Returns:
        {
            "indicators": {
                "FEDFUNDS": {"label": str, "latest_value": float, "latest_date": str},
                ...
            },
            "fetched_at": str,
            "from_cache": bool,
            "error": str | None  # エラー時のみ
        }
    """
    cache_key = "macro:indicators"
    cached = _get_cached(cache_key)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    result: dict[str, Any] = {
        "indicators": {},
        "fetched_at": datetime.now().isoformat(),
        "from_cache": False,
        "error": None,
    }

    if not FRED_API_KEY:
        result["error"] = "FRED_API_KEY が設定されていません"
        return result

    try:
        indicators = {}

        for series_id, label in MACRO_SERIES.items():
            try:
                resp = requests.get(
                    FRED_BASE_URL,
                    params={
                        "series_id": series_id,
                        "api_key": FRED_API_KEY,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 1,
                    },
                    timeout=10,
                )
                resp.raise_for_status()

                data = resp.json()

                if "observations" not in data or not data["observations"]:
                    indicators[series_id] = {
                        "label": label,
                        "latest_value": None,
                        "latest_date": None,
                        "error": "データなし",
                    }
                    continue

                obs = data["observations"][0]
                value_str = obs.get("value", ".")
                latest_date = obs.get("date")

                if value_str == ".":
                    latest_value = None
                else:
                    try:
                        latest_value = float(value_str)
                    except (ValueError, TypeError):
                        latest_value = None

                indicators[series_id] = {
                    "label": label,
                    "latest_value": latest_value,
                    "latest_date": latest_date,
                }

            except requests.RequestException as e:
                logger.warning("FRED API リクエスト失敗 (%s): %s", series_id, e)
                indicators[series_id] = {
                    "label": label,
                    "latest_value": None,
                    "latest_date": None,
                    "error": str(e),
                }

        result["indicators"] = indicators
        if result["error"] is None:
            _set_cached(cache_key, result)

    except Exception as e:
        logger.exception("マクロ指標取得中にエラーが発生しました")
        result["error"] = str(e)

    return result

if __name__ == "__main__":
    print("--- 市場データ取得テスト開始 ---")
    
    # 為替レートのテスト
    print("【為替レート取得中...】")
    print(get_exchange_rate("USDJPY=X"))
    
    # マクロ指標のテスト
    print("\n【マクロ指標取得中...】")
    print(get_macro_indicators())
    
    print("\n--- テスト完了 ---")