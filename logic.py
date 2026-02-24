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
    raise ValueError(
        "GEMINI_API_KEY が設定されていません。"
        ".env ファイルに GEMINI_API_KEY を追加してください。"
    )

if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    raise ValueError(
        "Langfuse の認証情報が不足しています。"
        ".env に LANGFUSE_PUBLIC_KEY と LANGFUSE_SECRET_KEY を設定してください。"
    )

# Gemini クライアント
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Langfuse クライアント（トレーシング用）
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)

# -----------------------------------------------------------------------------
# 出力スキーマ（IS-LM-BP 分析結果）
# -----------------------------------------------------------------------------
MACRO_IMPACT_SCHEMA = {
    "type": "object",
    "required": ["is_shift", "lm_shift", "bp_shift", "logic_jp"],
    "properties": {
        "is_shift": {
            "type": "string",
            "enum": ["right", "left", "none"],
            "description": "IS曲線のシフト方向",
        },
        "lm_shift": {
            "type": "string",
            "enum": ["right", "left", "none"],
            "description": "LM曲線のシフト方向",
        },
        "bp_shift": {
            "type": "string",
            "enum": ["upward", "downward", "none"],
            "description": "BP曲線のシフト方向",
        },
        "logic_jp": {
            "type": "string",
            "description": "日本語での詳細な経済学的解説",
        },
    },
}

SYSTEM_INSTRUCTION = (
    "あなたはマクロ経済学と外国為替市場の専門家です。"
    "入力されたニュースについて、マンデル＝フレミング・モデルにおける"
    "IS曲線・LM曲線・BP曲線への影響を分析し、指定されたJSON形式で出力してください。"
)

USER_PROMPT_TEMPLATE = """
以下のニュースを分析し、マンデル＝フレミング・モデルにおける
IS曲線・LM曲線・BP曲線がどの方向にシフトするかを判定してください。

【ニュース】
{news_text}

上記のJSON形式で、以下のルールに従って厳格に出力してください：
- is_shift: IS曲線のシフト（right / left / none）
- lm_shift: LM曲線のシフト（right / left / none）
- bp_shift: BP曲線のシフト（upward / downward / none）
- logic_jp: 日本語での詳細な経済学的解説。IS-LM-BPのどれがなぜ動いたか、利子率と為替への影響を説明してください。
"""

# -----------------------------------------------------------------------------
# 統合市場データ取得（経済指標カレンダー + NewsAPI）
# -----------------------------------------------------------------------------
TRADING_ECONOMICS_API_KEY = os.getenv("TRADING_ECONOMICS_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TE_BASE = "https://api.tradingeconomics.com"

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
            resp.raise_for_status()
            raw = resp.json()
            events = raw if isinstance(raw, list) else []
            result["economic_calendar"] = _sort_and_limit_events(events)
        except Exception as e:
            logger.warning("Trading Economics 取得失敗: %s", e)
            result["errors"].append(f"Trading Economics: {e}")
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

    parts.append("上記のJSON形式で、以下のルールに従って厳格に出力してください：")
    parts.append("- is_shift: IS曲線のシフト（right / left / none）")
    parts.append("- lm_shift: LM曲線のシフト（right / left / none）")
    parts.append("- bp_shift: BP曲線のシフト（upward / downward / none）")
    parts.append("- logic_jp: 日本語での詳細な経済学的解説。IS-LM-BPのどれがなぜ動いたか、利子率と為替への影響を説明してください。")
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
            "is_shift": "right" | "left" | "none",
            "lm_shift": "right" | "left" | "none",
            "bp_shift": "upward" | "downward" | "none",
            "logic_jp": "日本語での詳細な経済学的解説"
        }

    Raises:
        ValueError: news_text が空の場合
        RuntimeError: Gemini API 呼び出しに失敗した場合
    """
    if not news_text or not news_text.strip():
        raise ValueError("news_text が空です。分析対象のニュースを入力してください。")

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

    # 短命アプリでのトレース送信を確実にする
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

    langfuse.flush()

    return {
        "analysis": result,
        "economic_calendar": integrated_data.get("economic_calendar", []),
    }


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

