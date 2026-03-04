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
from logic import (
    FALLBACK_US_CPI_YOY,
    FALLBACK_US_POLICY_RATE,
    FALLBACK_USD_JPY,
    analyze_macro_impact_with_integrated_data,
    get_te_macro_snapshot,
)

import os
import uvicorn

from calendar_logic import get_today_economic_calendar

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
@app.get("/")
async def root() -> dict[str, str]:
    """トップページ：API 稼働確認用"""
    return {"message": "Mundel API is running"}


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """サーバーの稼働確認用"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/analysis")
async def get_analysis() -> dict[str, Any]:
    """
    アプリ起動時に上部カード（USD/JPY、金利、CPI）を埋めるための
    最新市場データを返す。
    """
    exchange_result, macro_result, te_snapshot_result = await asyncio.gather(
        asyncio.to_thread(get_exchange_rate, "USDJPY=X"),
        asyncio.to_thread(get_macro_indicators),
        asyncio.to_thread(get_te_macro_snapshot),
        return_exceptions=True,
    )

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

    if isinstance(te_snapshot_result, Exception):
        te_macro = {
            "usd_jpy": FALLBACK_USD_JPY,
            "us_policy_rate": FALLBACK_US_POLICY_RATE,
            "us_cpi_yoy": FALLBACK_US_CPI_YOY,
            "errors": [str(te_snapshot_result)],
        }
    else:
        te_macro = {
            "usd_jpy": te_snapshot_result.get("usd_jpy", FALLBACK_USD_JPY),
            "us_policy_rate": te_snapshot_result.get("us_policy_rate", FALLBACK_US_POLICY_RATE),
            "us_cpi_yoy": te_snapshot_result.get("us_cpi_yoy", FALLBACK_US_CPI_YOY),
            "usd_jpy_source": te_snapshot_result.get("usd_jpy_source"),
            "us_policy_rate_source": te_snapshot_result.get("us_policy_rate_source"),
            "us_cpi_yoy_source": te_snapshot_result.get("us_cpi_yoy_source"),
            "errors": te_snapshot_result.get("errors", []),
        }

    return {
        "market_data": market_data,
        "te_macro_snapshot": te_macro,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/analyze")
async def analyze(news: AnalyzeRequest) -> dict[str, Any]:
    """
    ニュースを分析し、AI分析結果と市場データを統合して返却する。

    - analyze_macro_impact と市場データ取得（為替・マクロ指標）を並行実行
    - いずれかが失敗しても、エラーを含めてレスポンスを返しクラッシュしない
    """
    news_text = news.news_text.strip()

    # 統合データを用いた分析 + 市場データ + TE マクロスナップショットを並行実行
    analysis_result, exchange_result, macro_result, te_snapshot_result = await asyncio.gather(
        asyncio.to_thread(analyze_macro_impact_with_integrated_data, news_text),
        asyncio.to_thread(get_exchange_rate, "USDJPY=X"),
        asyncio.to_thread(get_macro_indicators),
        asyncio.to_thread(get_te_macro_snapshot),
        return_exceptions=True,
    )

    # 分析結果の正規化（統合分析は analysis + economic_calendar を返す）
    economic_calendar: list = []
    if isinstance(analysis_result, Exception):
        analysis = {"error": str(analysis_result)}
    else:
        analysis = analysis_result.get("analysis", analysis_result)
        economic_calendar = analysis_result.get("economic_calendar", [])

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

    # Trading Economics マクロスナップショット（USD/JPY, 米政策金利, 米CPI YoY）
    te_macro: dict[str, Any] = {}
    if isinstance(te_snapshot_result, Exception):
        te_macro = {
            "usd_jpy": FALLBACK_USD_JPY,
            "us_policy_rate": FALLBACK_US_POLICY_RATE,
            "us_cpi_yoy": FALLBACK_US_CPI_YOY,
            "errors": [str(te_snapshot_result)],
        }
    else:
        te_macro = {
            "usd_jpy": te_snapshot_result.get("usd_jpy", FALLBACK_USD_JPY),
            "us_policy_rate": te_snapshot_result.get("us_policy_rate", FALLBACK_US_POLICY_RATE),
            "us_cpi_yoy": te_snapshot_result.get("us_cpi_yoy", FALLBACK_US_CPI_YOY),
            "usd_jpy_source": te_snapshot_result.get("usd_jpy_source"),
            "us_policy_rate_source": te_snapshot_result.get("us_policy_rate_source"),
            "us_cpi_yoy_source": te_snapshot_result.get("us_cpi_yoy_source"),
            "errors": te_snapshot_result.get("errors", []),
        }

    return {
        "analysis": analysis,
        "market_data": market_data,
        "economic_calendar": economic_calendar,
        "te_macro_snapshot": te_macro,
        "timestamp": datetime.now().isoformat(),
    }

if __name__ == "__main__":
    # 環境変数 PORT があればそれを使い、なければ 8000 を使う（ローカル用）
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.get("/api/calendar")
def read_calendar():
    return get_today_economic_calendar()