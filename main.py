"""
Mundel バックエンド API

FastAPI でフロントエンド（v0 / React / Next.js）と接続する。
ニュース分析と市場データを統合して返却する。
"""

import asyncio
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_fetcher import get_exchange_rate, get_macro_indicators
from logic import analyze_macro_impact

import os
import uvicorn

# -----------------------------------------------------------------------------
# FastAPI アプリケーション
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Mundel API",
    description="経済学ベースのFX分析API",
    version="0.1.0",
)

# v0 / React / Next.js からのアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では特定オリジンに制限推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# リクエスト・レスポンスモデル
# -----------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    """POST /api/analyze のリクエストボディ"""

    news_text: str = Field(..., min_length=1, description="分析対象のニューステキスト")


# -----------------------------------------------------------------------------
# エンドポイント
# -----------------------------------------------------------------------------
@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """サーバーの稼働確認用"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze")
async def analyze(news: AnalyzeRequest) -> dict[str, Any]:
    """
    ニュースを分析し、AI分析結果と市場データを統合して返却する。

    - analyze_macro_impact と市場データ取得（為替・マクロ指標）を並行実行
    - いずれかが失敗しても、エラーを含めてレスポンスを返しクラッシュしない
    """
    news_text = news.news_text.strip()

    # 3つの処理を並行実行（同期関数は to_thread で実行）
    analysis_result, exchange_result, macro_result = await asyncio.gather(
        asyncio.to_thread(analyze_macro_impact, news_text),
        asyncio.to_thread(get_exchange_rate, "USDJPY=X"),
        asyncio.to_thread(get_macro_indicators),
        return_exceptions=True,
    )

    # 分析結果の正規化（例外時は error 付きで格納）
    if isinstance(analysis_result, Exception):
        analysis = {"error": str(analysis_result)}
    else:
        analysis = analysis_result

    # 市場データの統合
    market_data: dict[str, Any] = {
        "exchange": {},
        "indicators": {},
        "errors": [],
    }

    if isinstance(exchange_result, Exception):
        market_data["errors"].append(f"為替データ: {exchange_result}")
    else:
        market_data["exchange"] = {
            "pair": exchange_result.get("pair", "USDJPY=X"),
            "current_price": exchange_result.get("current_price"),
            "closes_7d": exchange_result.get("closes_7d", []),
            "error": exchange_result.get("error"),
        }

    if isinstance(macro_result, Exception):
        market_data["errors"].append(f"マクロ指標: {macro_result}")
    else:
        indicators = macro_result.get("indicators", {})
        market_data["indicators"] = {
            "us_policy_rate": indicators.get("FEDFUNDS", {}).get("latest_value"),
            "us_cpi": indicators.get("CPIAUCSL", {}).get("latest_value"),
            "jp_policy_rate": indicators.get("IRSTCB01JPM156N", {}).get("latest_value"),
            "jp_cpi": indicators.get("JPNCPIALLMINMEI", {}).get("latest_value"),
            "raw": indicators,
        }
        if macro_result.get("error"):
            market_data["errors"].append(macro_result["error"])

    return {
        "analysis": analysis,
        "market_data": market_data,
        "timestamp": datetime.now().isoformat(),
    }

if __name__ == "__main__":
    # 環境変数 PORT があればそれを使い、なければ 8000 を使う（ローカル用）
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)