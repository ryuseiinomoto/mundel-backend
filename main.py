"""
Mundel バックエンド API

FastAPI でフロントエンド（v0 / React / Next.js）と接続する。
ニュース分析と市場データを統合して返却する。
"""

import asyncio
from datetime import datetime
import json
import logging
import sqlite3
from typing import Any
import uuid

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_fetcher import DATA_DIR, DB_PATH, get_exchange_rate, get_macro_indicators
from logic import (
    FALLBACK_US_CPI_YOY,
    FALLBACK_US_POLICY_RATE,
    FALLBACK_USD_JPY,
    analyze_macro_impact_with_integrated_data,
    compute_equilibrium,
    get_te_macro_snapshot,
    fx_chat_response,
)

import os
import uvicorn
import yfinance as yf

from calendar_logic import get_today_economic_calendar

# -----------------------------------------------------------------------------
# FastAPI アプリケーション
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Mundel API",
    description="経済学ベースのFX分析API",
    version="0.1.0",
)

# 許可するオリジンを環境変数で制御（カンマ区切り）
# 例: ALLOWED_ORIGINS=https://mundel.vercel.app,https://mundel-preview.vercel.app
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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


class EquilibriumPoint(BaseModel):
    """均衡点 (Y, r)"""

    y: float = Field(..., description="均衡所得")
    r: float = Field(..., description="均衡利子率")


class MacroEffects(BaseModel):
    """マクロ経済指標への予測影響"""

    exchange_rate: str = Field("Neutral", description="為替への影響")
    interest_rate: str = Field("Neutral", description="利子率への影響")
    output: str = Field("Neutral", description="産出・所得への影響")
    capital_flow: str = Field("Neutral", description="資本フローへの影響")


class AnalyzeResponse(BaseModel):
    """POST /api/analyze のレスポンス（フロントエンドが確実に受け取れるよう数値型を保証）"""

    analysis: dict[str, Any] = Field(default_factory=dict)
    market_data: dict[str, Any] = Field(default_factory=dict)
    economic_calendar: list = Field(default_factory=list)
    te_macro_snapshot: dict[str, Any] = Field(default_factory=dict)
    is_shift: float = Field(0.0, ge=-10.0, le=10.0, description="IS曲線シフト量")
    lm_shift: float = Field(0.0, ge=-10.0, le=10.0, description="LM曲線シフト量")
    bp_shift: float = Field(0.0, ge=-10.0, le=10.0, description="BP曲線シフト量")
    equilibrium_e0: EquilibriumPoint = Field(..., description="現在の均衡点")
    equilibrium_e1: EquilibriumPoint = Field(..., description="シフト後の均衡点")
    predicted_equilibrium: EquilibriumPoint = Field(..., description="予測均衡点（equilibrium_e1 と同じ）")
    shifts_delta: dict[str, float] = Field(
        ...,
        description="シフト量 (is, lm, bp)。フロントエンド互換のため is キーを使用",
    )
    macro_effects: MacroEffects = Field(
        default_factory=lambda: MacroEffects(),
        description="為替・利子率・産出・資本フローへの予測影響",
    )
    signal: str = Field("HOLD", description="売買シグナル: BUY | SELL | HOLD")
    signal_reason: str = Field("", description="シグナルの根拠（日本語）")
    timestamp: str = Field(default="")


# -----------------------------------------------------------------------------
# 共通ヘルパー: 市場データ整形
# -----------------------------------------------------------------------------
def _build_market_data(exchange_result: Any, macro_result: Any) -> dict[str, Any]:
    """為替・マクロ指標の取得結果を統一フォーマットに整形する"""
    market_data: dict[str, Any] = {"exchange": {}, "indicators": {}, "errors": []}

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

    return market_data


def _build_te_macro(te_snapshot_result: Any) -> dict[str, Any]:
    """Trading Economics スナップショットの取得結果を統一フォーマットに整形する"""
    if isinstance(te_snapshot_result, Exception):
        return {
            "usd_jpy": FALLBACK_USD_JPY,
            "us_policy_rate": FALLBACK_US_POLICY_RATE,
            "us_cpi_yoy": FALLBACK_US_CPI_YOY,
            "errors": [str(te_snapshot_result)],
        }
    return {
        "usd_jpy": te_snapshot_result.get("usd_jpy", FALLBACK_USD_JPY),
        "us_policy_rate": te_snapshot_result.get("us_policy_rate", FALLBACK_US_POLICY_RATE),
        "us_cpi_yoy": te_snapshot_result.get("us_cpi_yoy", FALLBACK_US_CPI_YOY),
        "usd_jpy_source": te_snapshot_result.get("usd_jpy_source"),
        "us_policy_rate_source": te_snapshot_result.get("us_policy_rate_source"),
        "us_cpi_yoy_source": te_snapshot_result.get("us_cpi_yoy_source"),
        "errors": te_snapshot_result.get("errors", []),
    }


# -----------------------------------------------------------------------------
# トレード状態（SQLite 永続化）
# -----------------------------------------------------------------------------
INITIAL_BALANCE = 10000.0


def _get_trade_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _load_trade_state() -> dict[str, Any]:
    try:
        conn = _get_trade_db()
        rows = dict(conn.execute("SELECT key, value FROM trade_state").fetchall())
        conn.close()
        return {
            "balance": float(rows.get("balance", INITIAL_BALANCE)),
            "positions": json.loads(rows.get("positions", "[]")),
        }
    except Exception as e:
        logger.warning("トレード状態の読み込みに失敗しました: %s", e)
        return {"balance": INITIAL_BALANCE, "positions": []}


def _save_trade_state(state: dict[str, Any]) -> None:
    try:
        conn = _get_trade_db()
        conn.execute(
            "INSERT OR REPLACE INTO trade_state (key, value) VALUES (?, ?)",
            ("balance", str(state["balance"])),
        )
        conn.execute(
            "INSERT OR REPLACE INTO trade_state (key, value) VALUES (?, ?)",
            ("positions", json.dumps(state["positions"], ensure_ascii=False)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("トレード状態の保存に失敗しました: %s", e)


class TradeRequest(BaseModel):
    action: str = Field(..., description="BUY または SELL")
    quantity: float = Field(..., gt=0, description="取引数量（USD）")


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

    return {
        "market_data": _build_market_data(exchange_result, macro_result),
        "te_macro_snapshot": _build_te_macro(te_snapshot_result),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(news: AnalyzeRequest) -> AnalyzeResponse:
    """
    ニュースを分析し、AI分析結果と市場データを統合して返却する。

    - analyze_macro_impact と市場データ取得（為替・マクロ指標）を並行実行
    - is_shift, lm_shift, bp_shift, predicted_equilibrium は必ず数値で返却
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
    market_data = _build_market_data(exchange_result, macro_result)
    te_macro = _build_te_macro(te_snapshot_result)

    # 均衡点 E0（現在）と E1（シフト後予測）を算出（必ず数値で返す）
    def to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    def clamp(v: float) -> float:
        return max(-10.0, min(10.0, v))

    is_d = clamp(to_float(analysis.get("is_shift"), 0.0))
    lm_d = clamp(to_float(analysis.get("lm_shift"), 0.0))
    bp_d = clamp(to_float(analysis.get("bp_shift"), 0.0))
    eq0 = compute_equilibrium(0.0, 0.0, 0.0)
    eq1 = compute_equilibrium(is_d, lm_d, bp_d)

    # macro_effects: AI の分析結果から取得、なければシフト量から推定
    def _str(v: Any, default: str = "Neutral") -> str:
        s = str(v).strip() if v is not None else ""
        return s if s else default

    def _derive_from_shifts(iso: float, lmo: float, bpo: float) -> MacroEffects:
        """IS-LM-BP シフトからマクロ影響を簡易推定"""
        y_delta = iso - lmo
        return MacroEffects(
            exchange_rate="Appreciation" if bpo > 0.5 else ("Depreciation" if bpo < -0.5 else "Neutral"),
            interest_rate="Increase" if iso > 0.5 or lmo < -0.5 else ("Decrease" if iso < -0.5 or lmo > 0.5 else "Neutral"),
            output="Expand" if y_delta > 0.5 else ("Contract" if y_delta < -0.5 else "Neutral"),
            capital_flow="Inflow" if bpo > 0.5 else ("Outflow" if bpo < -0.5 else "Neutral"),
        )

    me_raw = analysis.get("macro_effects") or {}
    if isinstance(me_raw, dict) and any(me_raw.get(k) for k in ("exchange_rate", "interest_rate", "output", "capital_flow")):
        me = MacroEffects(
            exchange_rate=_str(me_raw.get("exchange_rate")),
            interest_rate=_str(me_raw.get("interest_rate")),
            output=_str(me_raw.get("output")),
            capital_flow=_str(me_raw.get("capital_flow")),
        )
    else:
        me = _derive_from_shifts(is_d, lm_d, bp_d)

    signal = str(analysis.get("signal") or "HOLD").upper()
    if signal not in ("BUY", "SELL", "HOLD"):
        signal = "HOLD"
    signal_reason = str(analysis.get("signal_reason") or "")

    return AnalyzeResponse(
        analysis=analysis,
        market_data=market_data,
        economic_calendar=economic_calendar,
        te_macro_snapshot=te_macro,
        is_shift=is_d,
        lm_shift=lm_d,
        bp_shift=bp_d,
        equilibrium_e0=EquilibriumPoint(y=eq0["y"], r=eq0["r"]),
        equilibrium_e1=EquilibriumPoint(y=eq1["y"], r=eq1["r"]),
        predicted_equilibrium=EquilibriumPoint(y=eq1["y"], r=eq1["r"]),
        shifts_delta={"is": is_d, "lm": lm_d, "bp": bp_d},
        macro_effects=me,
        signal=signal,
        signal_reason=signal_reason,
        timestamp=datetime.now().isoformat(),
    )

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="FXに関する質問・相談")


@app.post("/api/chat")
async def chat_fx(req: ChatRequest) -> dict[str, str]:
    """FXに関する質問・相談に回答する汎用チャットエンドポイント"""
    answer = await asyncio.to_thread(fx_chat_response, req.message.strip())
    return {"answer": answer}


@app.get("/api/calendar")
def read_calendar():
    return get_today_economic_calendar()


# -----------------------------------------------------------------------------
# トレードエンドポイント
# -----------------------------------------------------------------------------
@app.get("/api/trade/chart")
async def get_trade_chart() -> dict[str, Any]:
    """USD/JPY の直近2日分・5分足OHLCデータを返す（最大100本）"""
    try:
        ticker = yf.Ticker("USDJPY=X")
        hist = ticker.history(period="2d", interval="5m")
        candles = []
        for ts, row in hist.iterrows():
            candles.append({
                "date": ts.strftime("%m/%d %H:%M"),
                "open": round(float(row["Open"]), 3),
                "high": round(float(row["High"]), 3),
                "low": round(float(row["Low"]), 3),
                "close": round(float(row["Close"]), 3),
            })
        # 最新100本のみ返す
        candles = candles[-100:]
        current_price = candles[-1]["close"] if candles else FALLBACK_USD_JPY
        return {"candles": candles, "current_price": current_price}
    except Exception as e:
        return {"candles": [], "current_price": FALLBACK_USD_JPY, "error": str(e)}


@app.get("/api/trade")
async def get_trade_state() -> dict[str, Any]:
    """現在のトレード状態（残高・ポジション）を返す"""
    state = _load_trade_state()
    current_price = FALLBACK_USD_JPY
    try:
        fx = await asyncio.to_thread(get_exchange_rate, "USDJPY=X")
        p = fx.get("current_price")
        if isinstance(p, (int, float)):
            current_price = float(p)
    except Exception:
        pass

    positions_with_pnl = []
    for pos in state["positions"]:
        entry = pos["entry_price"]
        qty = pos["quantity"]
        if pos["action"] == "BUY":
            pnl = (current_price - entry) * qty / entry
        else:
            pnl = (entry - current_price) * qty / entry
        positions_with_pnl.append({**pos, "current_price": current_price, "pnl": round(pnl, 2)})

    total_pnl = sum(p["pnl"] for p in positions_with_pnl)

    return {
        "balance": round(state["balance"], 2),
        "positions": positions_with_pnl,
        "total_pnl": round(total_pnl, 2),
        "current_price": current_price,
    }


@app.post("/api/trade")
async def execute_trade(req: TradeRequest) -> dict[str, Any]:
    """模擬トレードを実行する（BUY / SELL）"""
    action = req.action.upper()
    if action not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="action は BUY または SELL を指定してください")

    # 現在価格取得
    current_price = FALLBACK_USD_JPY
    try:
        fx = await asyncio.to_thread(get_exchange_rate, "USDJPY=X")
        p = fx.get("current_price")
        if isinstance(p, (int, float)):
            current_price = float(p)
    except Exception:
        pass

    state = _load_trade_state()
    cost = req.quantity  # USD建て
    if cost > state["balance"]:
        raise HTTPException(status_code=400, detail=f"残高不足です。残高: ${state['balance']:.2f}")

    state["balance"] -= cost
    position = {
        "id": str(uuid.uuid4())[:8],
        "action": action,
        "quantity": req.quantity,
        "entry_price": current_price,
        "entry_time": datetime.now().isoformat(),
    }
    state["positions"].append(position)
    _save_trade_state(state)

    return {
        "message": f"{action} {req.quantity} USD @ {current_price:.3f} JPY を執行しました",
        "position": position,
        "balance": round(state["balance"], 2),
    }


@app.delete("/api/trade/{position_id}")
async def close_position(position_id: str) -> dict[str, Any]:
    """ポジションをクローズして損益を確定する"""
    state = _load_trade_state()
    pos_idx = next((i for i, p in enumerate(state["positions"]) if p["id"] == position_id), None)
    if pos_idx is None:
        raise HTTPException(status_code=404, detail="指定されたポジションが見つかりません")

    # 現在価格取得
    current_price = FALLBACK_USD_JPY
    try:
        fx = await asyncio.to_thread(get_exchange_rate, "USDJPY=X")
        p = fx.get("current_price")
        if isinstance(p, (int, float)):
            current_price = float(p)
    except Exception:
        pass

    pos = state["positions"].pop(pos_idx)
    entry = pos["entry_price"]
    qty = pos["quantity"]

    if pos["action"] == "BUY":
        pnl = (current_price - entry) * qty / entry
    else:
        pnl = (entry - current_price) * qty / entry

    state["balance"] += qty + pnl  # 証拠金返却 + 損益
    _save_trade_state(state)

    return {
        "message": f"ポジション {position_id} をクローズしました",
        "pnl": round(pnl, 2),
        "close_price": current_price,
        "balance": round(state["balance"], 2),
    }


@app.post("/api/trade/reset")
async def reset_trade() -> dict[str, Any]:
    """トレード状態をリセットする（初期残高に戻す）"""
    _save_trade_state({"balance": INITIAL_BALANCE, "positions": []})
    return {"message": "トレード状態をリセットしました", "balance": INITIAL_BALANCE}


if __name__ == "__main__":
    # 環境変数 PORT があればそれを使い、なければ 8000 を使う（ローカル用）
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)