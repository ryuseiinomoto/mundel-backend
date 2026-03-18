"""
Mundel 心臓部: マクロ経済ニュースのIS-LM-BP分析ロジック

Gemini API でニュースを分析し、マンデル＝フレミング・モデルに基づく
IS曲線・LM曲線・BP曲線への影響を判定する。
経済指標カレンダー（Trading Economics：米国・日本）と NewsAPI（市場ニュース）の統合データも利用する。
Langfuse で分析過程をトレースする。
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langfuse import Langfuse, observe
import requests
from newsapi import NewsApiClient
from data_fetcher import get_exchange_rate

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 環境変数の読み込み
# -----------------------------------------------------------------------------
# .env はプロジェクトルート（mundel-backend の親）に配置
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# -----------------------------------------------------------------------------
# API クライアントの初期化（エラーハンドリング付き）
# -----------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY が設定されていません。"
        ".env ファイルに GEMINI_API_KEY を追加してください。"
    )

if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    logger.warning(
        "Langfuse の認証情報が不足しています。"
        ".env に LANGFUSE_PUBLIC_KEY と LANGFUSE_SECRET_KEY を設定してください。"
        "トレーシングは無効化されます。"
    )

# Gemini クライアント（キー未設定時は None、呼び出し時にエラー）
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Langfuse クライアント（認証情報未設定時は None）
langfuse = (
    Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
    else None
)

# -----------------------------------------------------------------------------
# 出力スキーマ（IS-LM-BP 分析結果）
# -----------------------------------------------------------------------------
MACRO_IMPACT_SCHEMA = {
    "type": "object",
    "required": ["is_shift", "lm_shift", "bp_shift", "logic_jp", "signal", "signal_reason"],
    "properties": {
        "is_shift": {
            "type": "number",
            "description": "IS曲線の予測シフト量。正=右シフト、負=左シフト。範囲 -10.0 〜 10.0",
        },
        "lm_shift": {
            "type": "string",
            "description": "LM曲線の方向。'left'=左シフト（利上げ・金融引き締め）、'right'=右シフト（利下げ・金融緩和）、'neutral'=変化なし",
            "enum": ["left", "right", "neutral"],
        },
        "lm_shift_magnitude": {
            "type": "number",
            "description": "LM曲線シフトの大きさ（0〜10）。任意。省略時は5を使用",
        },
        "bp_shift": {
            "type": "number",
            "description": "BP曲線の予測シフト量。正=上シフト、負=下シフト。範囲 -10.0 〜 10.0",
        },
        "logic_jp": {
            "type": "string",
            "description": "日本語での詳細な経済学的解説",
        },
        "macro_effects": {
            "type": "object",
            "description": "マクロ経済指標への予測影響。英語で簡潔に。",
            "properties": {
                "exchange_rate": {"type": "string", "description": "為替への影響 e.g. Appreciation, Depreciation, Neutral"},
                "interest_rate": {"type": "string", "description": "利子率への影響 e.g. Increase, Decrease, Neutral"},
                "output": {"type": "string", "description": "産出・所得への影響 e.g. Expand, Contract, Neutral"},
                "capital_flow": {"type": "string", "description": "資本フローへの影響 e.g. Inflow, Outflow, Neutral"},
            },
        },
        "signal": {
            "type": "string",
            "description": "USD/JPYの売買シグナル。BUY=ドル買い・円売り（円安予測）、SELL=ドル売り・円買い（円高予測）、HOLD=様子見",
            "enum": ["BUY", "SELL", "HOLD"],
        },
        "signal_reason": {
            "type": "string",
            "description": "シグナルの根拠（日本語で簡潔に）。IS-LM-BPのシフト方向と為替への影響を踏まえた判断理由。",
        },
    },
}

SYSTEM_INSTRUCTION = (
    "あなたはマクロ経済学と外国為替市場の専門家です。"
    "入力されたニュースについて、マンデル＝フレミング・モデルにおける"
    "IS曲線・LM曲線・BP曲線への影響を分析し、指定されたJSON形式で出力してください。"
    "【重要】LM曲線: FRB利上げ・金融引き締めのときは必ず 'left'（左シフト）。利下げ・金融緩和のときは 'right'（右シフト）。"
    "【シグナル判定ルール】USD/JPYの売買シグナルを以下のロジックで判定してください："
    "BUY（ドル買い・円安予測）: IS右シフト＋BP上シフト→資本流入・円安圧力、またはFRB利上げ→日米金利差拡大→円安。"
    "SELL（ドル売り・円高予測）: IS左シフト＋BP下シフト→資本流出・円高圧力、またはFRB利下げ→日米金利差縮小→円高、またはBOJ利上げ期待→円高。"
    "HOLD: IS・LM・BP変化が小さい、または相反するシグナルで判断困難なとき。"
)

USER_PROMPT_TEMPLATE = """
以下のニュースを分析し、マンデル＝フレミング・モデルにおける
IS曲線・LM曲線・BP曲線の予測シフト量を数値で判定してください。

【ニュース】
{news_text}

【返すべきJSON形式】
必ず以下の形式のJSONオブジェクトを1つだけ出力してください。各フィールドは数値または文字列で、型を厳守してください。

{{
  "is_shift": 数値,
  "lm_shift": "left" | "right" | "neutral",
  "bp_shift": 数値,
  "logic_jp": "日本語での経済学的解説",
  "macro_effects": {{ ... }}
}}

【ルール】
- is_shift: 数値（-10.0〜10.0）。右シフト=正、左シフト=負、なし=0。
- lm_shift: 必ず "left" または "right" または "neutral" のいずれか。
  - "left" = LM曲線が左（上）にシフト。FRB利上げ・金融引き締め・マネーサプライ減少のとき。
  - "right" = LM曲線が右（下）にシフト。利下げ・金融緩和・マネーサプライ増加のとき。
  - "neutral" = 変化なし。
- bp_shift: 数値（-10.0〜10.0）。上シフト=正、下シフト=負、なし=0。
- logic_jp: 日本語で説明。
- macro_effects: 必ず含める。
"""


def compute_equilibrium(is_delta: float, lm_delta: float, _bp_delta: float) -> dict[str, float]:
    """
    IS-LM モデルにおける均衡点 (Y*, r*) を算出する。
    フロントエンドの buildCurves と同じ係数を使用。
    BP は IS-LM 交点には影響しないため未使用。

    Returns:
        {"y": float, "r": float}  # Y軸=所得、r軸=利子率
    """
    # IS: r = 7 - 0.05Y + is*0.7
    # LM: r = 3 + 0.05Y + lm*0.7
    # 交点: 7 - 0.05Y + is*0.7 = 3 + 0.05Y + lm*0.7
    # 4 + 0.7*(is - lm) = 0.1Y  =>  Y = 40 + 7*(is - lm)
    y_eq = 40.0 + 7.0 * (is_delta - lm_delta)
    y_eq = max(0.0, min(100.0, y_eq))
    r_eq = 7.0 - 0.05 * y_eq + is_delta * 0.7
    r_eq = max(0.0, min(10.0, r_eq))
    return {"y": round(y_eq, 2), "r": round(r_eq, 2)}

# -----------------------------------------------------------------------------
# 統合市場データ取得（経済指標カレンダー + NewsAPI）
# -----------------------------------------------------------------------------
TRADING_ECONOMICS_API_KEY = os.getenv("TRADING_ECONOMICS_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TE_BASE = "https://api.tradingeconomics.com"

# フォールバック値（Trading Economics API 取得失敗時）
FALLBACK_USD_JPY = 150.0
FALLBACK_US_POLICY_RATE = 4.0
FALLBACK_US_CPI_YOY = 3.0


def get_te_macro_snapshot() -> dict[str, Any]:
    """
    以下を取得する:
    - 最新の USD/JPY レート（yfinance 経由）
    - アメリカの政策金利（Fed Funds Rate）
    - アメリカの CPI（前年比 / YoY）

    取得失敗時はフォールバック値を返す。
    """
    result: dict[str, Any] = {
        "usd_jpy": FALLBACK_USD_JPY,
        "us_policy_rate": FALLBACK_US_POLICY_RATE,
        "us_cpi_yoy": FALLBACK_US_CPI_YOY,
        "usd_jpy_source": None,
        "us_policy_rate_source": None,
        "us_cpi_yoy_source": None,
        "errors": [],
    }

    # USD/JPY: yfinance（data_fetcher.get_exchange_rate）で取得
    try:
        fx = get_exchange_rate("USDJPY=X")
        price = fx.get("current_price")
        if isinstance(price, (int, float)):
            result["usd_jpy"] = float(price)
            result["usd_jpy_source"] = "yfinance"
        elif fx.get("error"):
            result["errors"].append(f"USD/JPY (yfinance): {fx['error']}")
    except Exception as e:
        logger.warning("yfinance USD/JPY 取得失敗: %s", e)
        result["errors"].append(f"USD/JPY (yfinance): {e}")

    # 以下、金利・CPI は Trading Economics を継続利用

    if not TRADING_ECONOMICS_API_KEY:
        # TE キーが無くても、USD/JPY は yfinance で取得済みなのでそのまま返す
        result["errors"].append("TRADING_ECONOMICS_API_KEY が未設定です")
        return result

    params = {"c": TRADING_ECONOMICS_API_KEY}

    # 米国指標: country/united states
    try:
        url = f"{TE_BASE}/country/united%20states"
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning("Trading Economics 米国指標: HTTP %s", resp.status_code)
            result["errors"].append(f"US indicators: HTTP {resp.status_code}")
        else:
            raw = resp.json()
            items = raw if isinstance(raw, list) else []
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                cat = (item.get("Category") or "").lower()
                title = (item.get("Title") or "").lower()
                val = item.get("LatestValue")
                if val is None:
                    continue
                try:
                    num_val = float(val)
                except (TypeError, ValueError):
                    continue
                if "interest rate" in cat or "interest rate" in title or "fed funds" in cat or "fed funds" in title:
                    result["us_policy_rate"] = num_val
                    result["us_policy_rate_source"] = "Trading Economics"
                if (
                    ("consumer price index" in cat and ("yoy" in cat or "year" in cat))
                    or ("inflation" in cat and "cpi" in title)
                    or "consumer price index yoy" in cat
                ):
                    result["us_cpi_yoy"] = num_val
                    result["us_cpi_yoy_source"] = "Trading Economics"
            if result["us_cpi_yoy_source"] is None:
                for item in items or []:
                    if not isinstance(item, dict):
                        continue
                    cat = (item.get("Category") or "").lower()
                    val = item.get("LatestValue")
                    if val is None:
                        continue
                    try:
                        num_val = float(val)
                    except (TypeError, ValueError):
                        continue
                    if "inflation" in cat:
                        result["us_cpi_yoy"] = num_val
                        result["us_cpi_yoy_source"] = "Trading Economics"
                        break
    except Exception as e:
        logger.warning("Trading Economics 米国指標取得失敗: %s", e)
        result["errors"].append(f"US indicators: {e}")

    return result


def _sort_and_limit_events(events: list) -> list:
    """重要度（Importance: 3=高, 2=中, 1=低）でソートし、最大30件に制限"""
    def key(e):
        imp = e.get("Importance") or 0
        try:
            imp = int(imp)
        except (TypeError, ValueError):
            imp = 0
        date_str = e.get("Date") or ""
        return (-imp, date_str)
    return sorted(events, key=key)[:30]


def get_integrated_market_data() -> dict[str, Any]:
    """
    経済指標カレンダー（Trading Economics：米国・日本）と NewsAPI（市場ニュース）からデータを取得し統合する。

    Returns:
        {
            "economic_calendar": [...],  # 経済指標カレンダー（生データ）
            "news": [...],               # NewsAPI の市場ニュース
            "errors": [...]              # 取得失敗時のエラーメッセージ
        }
    """
    result: dict[str, Any] = {
        "economic_calendar": [],
        "news": [],
        "errors": [],
    }

    # Trading Economics: 経済指標カレンダー（米国・日本の重要指標）
    if TRADING_ECONOMICS_API_KEY:
        try:
            today = datetime.now()
            start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
            end = (today + timedelta(days=14)).strftime("%Y-%m-%d")
            url = f"{TE_BASE}/calendar/country/united%20states,japan"
            resp = requests.get(
                url,
                params={
                    "c": TRADING_ECONOMICS_API_KEY,
                    "from": start,
                    "to": end,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning("Trading Economics カレンダー: HTTP %s", resp.status_code)
                result["errors"].append(f"Trading Economics: HTTP {resp.status_code}")
            else:
                raw = resp.json()
                events = raw if isinstance(raw, list) else []
                result["economic_calendar"] = _sort_and_limit_events(events)
        except Exception as e:
            logger.warning("Trading Economics 取得失敗: %s", e)
            result["errors"].append(f"Trading Economics: {e}")
            # economic_calendar は初期値 [] のまま（クラッシュしない）
    else:
        result["errors"].append("TRADING_ECONOMICS_API_KEY が未設定です")

    # NewsAPI: 市場ニュース
    if NEWS_API_KEY:
        try:
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            headlines = newsapi.get_everything(
                q="USD JPY OR forex OR 為替 OR 金利 OR Fed OR 雇用統計 OR CPI",
                language="en",
                sort_by="publishedAt",
                page_size=10,
                from_param=from_date,
            )
            result["news"] = headlines.get("articles", []) or []
        except Exception as e:
            logger.warning("NewsAPI 取得失敗: %s", e)
            result["errors"].append(f"NewsAPI: {e}")
    else:
        result["errors"].append("NEWS_API_KEY が未設定です")

    return result


def generate_analysis_prompt(
    news_text: str,
    integrated_data: dict[str, Any],
) -> str:
    """
    統合市場データ（経済指標・ニュース）とユーザー入力ニュースを整形し、
    Gemini に渡すプロンプトを生成する。

    Args:
        news_text: ユーザーが入力したニューステキスト
        integrated_data: get_integrated_market_data() の戻り値

    Returns:
        Gemini 用のプロンプト文字列
    """
    parts = ["以下の情報を踏まえ、マンデル＝フレミング・モデルにおけるIS曲線・LM曲線・BP曲線への影響を分析し、指定されたJSON形式で出力してください。\n"]

    # ユーザー入力ニュース
    parts.append("【分析対象のニュース】")
    parts.append(news_text.strip())
    parts.append("")

    # Trading Economics から取得した最新の経済指標データ（米国・日本）
    calendar = integrated_data.get("economic_calendar", [])
    if calendar:
        parts.append("【Trading Economics から取得した最新の経済指標データ（米国・日本）】")
        for i, item in enumerate(calendar[:15], 1):
            if isinstance(item, dict):
                country = item.get("Country", "")
                category = item.get("Category", "")
                event = item.get("Event", "")
                date = (item.get("Date") or "")[:16]
                actual = item.get("Actual", "-")
                prev = item.get("Previous", "-")
                forecast = item.get("Forecast", "-")
                imp = item.get("Importance", "")
                parts.append(f"{i}. [{country}] {event or category} | {date} | Actual: {actual} | Prev: {prev} | Fcast: {forecast} (Importance: {imp})")
            else:
                parts.append(f"{i}. {item}")
        parts.append("")

    # NewsAPI ニュース
    news = integrated_data.get("news", [])
    if news:
        parts.append("【市場ニュース（NewsAPI）】")
        for i, article in enumerate(news[:8], 1):
            title = article.get("title", "")
            desc = article.get("description", "") or ""
            date = article.get("publishedAt", "")[:10] if article.get("publishedAt") else ""
            parts.append(f"{i}. [{date}] {title}")
            if desc:
                parts.append(f"   {desc[:150]}...")
        parts.append("")

    parts.append("【返すべきJSON形式】")
    parts.append("必ず以下の形式のJSONオブジェクトを1つだけ出力してください。")
    parts.append('{"is_shift": 数値, "lm_shift": "left"|"right"|"neutral", "bp_shift": 数値, "logic_jp": "解説文", "macro_effects": {...}}')
    parts.append("")
    parts.append("【ルール】")
    parts.append("- is_shift: 数値（-10〜10）。右=正、左=負、なし=0。")
    parts.append("- lm_shift: 必ず \"left\" または \"right\" または \"neutral\"。FRB利上げ・金融引き締め→\"left\"、利下げ・金融緩和→\"right\"。")
    parts.append("- bp_shift: 数値（-10〜10）。上=正、下=負、なし=0。")
    parts.append("- logic_jp: 日本語での詳細な経済学的解説。IS-LM-BPのどれがなぜ動いたか、利子率と為替への影響を説明。")
    parts.append("- macro_effects: exchange_rate, interest_rate, output, capital_flow を英語で簡潔に（Appreciation/Depreciation/Neutral, Increase/Decrease/Neutral, Expand/Contract/Neutral, Inflow/Outflow/Neutral）。")
    parts.append("- signal: \"BUY\"（ドル買い・円安予測）、\"SELL\"（ドル売り・円高予測）、\"HOLD\"（様子見）のいずれか。")
    parts.append("  BUY条件: IS右シフト＋BP上シフト（資本流入→円安）、またはFRB利上げ（日米金利差拡大→円安）。")
    parts.append("  SELL条件: IS左シフト＋BP下シフト（資本流出→円高）、またはFRB利下げ（金利差縮小→円高）、またはBOJ利上げ（円高）。")
    parts.append("  HOLD条件: シフト量が小さい（絶対値<1）または相反するシグナル。")
    parts.append("- signal_reason: シグナルの根拠を日本語で1〜2文で。")
    return "\n".join(parts)


@observe()
def analyze_macro_impact(news_text: str) -> dict:
    """
    マクロ経済ニュースを分析し、マンデル＝フレミング・モデルにおける
    IS曲線・LM曲線・BP曲線への影響を返す。

    Args:
        news_text: 分析対象のニューステキスト

    Returns:
        {
            "is_shift": float,  # -10.0 〜 10.0
            "lm_shift": float,
            "bp_shift": float,
            "logic_jp": str
        }

    Raises:
        ValueError: news_text が空の場合
        RuntimeError: Gemini API 呼び出しに失敗した場合
    """
    if not news_text or not news_text.strip():
        raise ValueError("news_text が空です。分析対象のニュースを入力してください。")

    if not genai_client:
        raise RuntimeError("GEMINI_API_KEY が設定されていないため分析できません。")

    user_prompt = USER_PROMPT_TEMPLATE.format(news_text=news_text.strip())

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=MACRO_IMPACT_SCHEMA,
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API の呼び出しに失敗しました: {e}") from e

    raw_text = response.text
    if not raw_text:
        raise RuntimeError("Gemini API が空の応答を返しました。")

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini API の応答をJSONとして解析できませんでした: {e}\n"
            f"応答内容: {raw_text[:500]}..."
        ) from e

    # スキーマ検証（必須キーの存在確認）
    required_keys = {"is_shift", "lm_shift", "bp_shift", "logic_jp"}
    if not required_keys.issubset(result.keys()):
        missing = required_keys - set(result.keys())
        raise RuntimeError(
            f"Gemini API の応答に必須キーが含まれていません: {missing}\n"
            f"応答内容: {result}"
        )

    def clamp_shift(v: Any) -> float:
        try:
            n = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(-10.0, min(10.0, n))

    def lm_direction_to_numeric(lm_val: Any) -> float:
        """LM曲線: left=正（左シフト）、right=負（右シフト）、neutral=0"""
        s = str(lm_val).strip().lower() if lm_val is not None else ""
        mag = clamp_shift(result.get("lm_shift_magnitude", 5))
        mag = max(0.5, mag)
        if s == "left":
            return mag
        if s == "right":
            return -mag
        return 0.0

    result["is_shift"] = clamp_shift(result.get("is_shift", 0))
    result["lm_shift"] = lm_direction_to_numeric(result.get("lm_shift"))
    result["bp_shift"] = clamp_shift(result.get("bp_shift", 0))

    # 短命アプリでのトレース送信を確実にする
    if langfuse:
        langfuse.flush()

    return result


@observe()
def analyze_macro_impact_with_integrated_data(news_text: str) -> dict[str, Any]:
    """
    統合市場データ（Twelve Data + NewsAPI）を用いて、
    Gemini による高度な FX 分析を行う。

    Returns:
        {
            "analysis": { is_shift, lm_shift, bp_shift, logic_jp },
            "economic_calendar": [...],  # Twelve Data の生データ（フロント表示用）
        }
    """
    if not news_text or not news_text.strip():
        raise ValueError("news_text が空です。分析対象のニュースを入力してください。")

    if not genai_client:
        raise RuntimeError("GEMINI_API_KEY が設定されていないため分析できません。")

    integrated_data = get_integrated_market_data()
    user_prompt = generate_analysis_prompt(news_text.strip(), integrated_data)

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=MACRO_IMPACT_SCHEMA,
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API の呼び出しに失敗しました: {e}") from e

    raw_text = response.text
    if not raw_text:
        raise RuntimeError("Gemini API が空の応答を返しました。")

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini API の応答をJSONとして解析できませんでした: {e}\n"
            f"応答内容: {raw_text[:500]}..."
        ) from e

    required_keys = {"is_shift", "lm_shift", "bp_shift", "logic_jp"}
    if not required_keys.issubset(result.keys()):
        missing = required_keys - set(result.keys())
        raise RuntimeError(
            f"Gemini API の応答に必須キーが含まれていません: {missing}\n"
            f"応答内容: {result}"
        )

    def clamp_shift(v: Any) -> float:
        try:
            n = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(-10.0, min(10.0, n))

    def lm_direction_to_numeric(lm_val: Any) -> float:
        """LM曲線: left=正（左シフト）、right=負（右シフト）、neutral=0"""
        s = str(lm_val).strip().lower() if lm_val is not None else ""
        mag = clamp_shift(result.get("lm_shift_magnitude", 5))
        mag = max(0.5, mag)  # 最小0.5
        if s == "left":
            return mag
        if s == "right":
            return -mag
        return 0.0

    result["is_shift"] = clamp_shift(result.get("is_shift", 0))
    result["lm_shift"] = lm_direction_to_numeric(result.get("lm_shift"))
    result["bp_shift"] = clamp_shift(result.get("bp_shift", 0))

    if langfuse:
        langfuse.flush()

    return {
        "analysis": result,
        "economic_calendar": integrated_data.get("economic_calendar", []),
    }


def fx_chat_response(message: str) -> str:
    """
    FXに関する質問・相談に日本語で回答する。
    """
    system = (
        "あなたはFX（外国為替取引）の専門家アドバイザーです。"
        "ユーザーのFXに関するあらゆる質問・相談に日本語でわかりやすく答えてください。"
        "具体的な数値や例を挙げながら説明し、初心者にも理解できるよう丁寧に解説してください。"
        "USD/JPYを中心に、相場の読み方、ローソク足の見方、トレード手法、"
        "リスク管理、経済指標の見方、IS-LM-BPモデルの説明なども対応してください。"
        "回答は400字以内で簡潔にまとめてください。"
    )
    if not genai_client:
        return "GEMINI_API_KEY が設定されていないため回答できません。"
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.5,
                max_output_tokens=600,
            ),
        )
        if langfuse:
            langfuse.flush()
        return response.text or "回答を生成できませんでした。"
    except Exception as e:
        logger.warning("fx_chat_response failed: %s", e)
        return f"申し訳ありません。エラーが発生しました。"


# --- ここからテスト用コード ---
if __name__ == "__main__":
    # 動作確認用のサンプルニュース
    test_news = "米国連邦準備制度（Fed）が、インフレ抑制のために0.5%の予想外の利上げを決定しました。"
    
    print("--- Mundel AI 分析開始 ---")
    try:
        # ステップ2で作った関数を呼び出す
        result = analyze_macro_impact(test_news)
        print("【分析結果】")
        print(result)
    except Exception as e:
        print(f"【エラー発生】: {e}")
    
    # Langfuseにデータを確実に送信するために必要
    langfuse.flush()
    print("--- テスト完了。Langfuseを確認してください ---")

