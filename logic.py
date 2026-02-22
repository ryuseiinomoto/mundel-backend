"""
Mundel 心臓部: マクロ経済ニュースのIS-LM-BP分析ロジック

Gemini API でニュースを分析し、マンデル＝フレミング・モデルに基づく
IS曲線・LM曲線・BP曲線への影響を判定する。
Langfuse で分析過程をトレースする。
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langfuse import Langfuse, observe

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

