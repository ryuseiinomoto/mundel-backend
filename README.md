Mundel のマクロ経済分析ロジックを支える FastAPI ベースの解析エンジンです。

## 概要
為替データおよび経済指標を取得し、マンデル＝フレミング・モデル（IS-LM-BP）に基づいた市場分析と AI によるインサイト生成を行います。

## 主な機能
- **Real-time Data Fetching**: `yfinance` を用いた最新為替レートの取得。
- **Economic Analysis**: 経済指標に基づいた IS/LM/BP 曲線のシフトロジックの実装。
- **AI Insights**: Google Gemini Pro を活用した、マクロ経済的観点からの為替予測と要約。
- **Observability**: `Langfuse` を統合し、AI の推論プロセスとトークン使用量をモニタリング。

## 技術スタック
- **Framework**: FastAPI (Python 3.12+)
- **Analysis**: Google Gemini Pro (Vertex AI / Generative AI)
- **Monitoring**: Langfuse
- **Data Source**: yfinance, Trading Economics (External API)
- **Deployment**: Google Cloud Run

## セットアップ
1. ライブラリのインストール
   ```bash
   pip install -r requirements.txt
