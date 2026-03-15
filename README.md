経済学のロジックと AI（Gemini Pro）を融合させた、Mundel の心臓部。

##  Core Logic
本システムは、単なる AI チャットボットではありません。
1. 外部 API から為替・経済指標を取得。
2. 経済ロジックに基づき、グラフの「シフト量」を計算。
3. AI がその背景をマクロ経済学的に解説。

##  Key Features & Engineering
- **AI Observability with Langfuse**:
  - `Langfuse` を統合し、AI の推論プロセス、トークン消費、応答時間を可視化。プロダクションレベルの監視体制を意識。
- **Fault Tolerant Architecture**:
  - **Exception Handling**: 外部 API（Trading Economics 等）がダウンしている際も、システム全体をクラッシュさせない `try-except` ロジックを実装。
- **FastAPI Performance**:
  - Python の非同期処理（`async/await`）を活用し、複数 API からの並列データ取得を実現。

##  Tech Stack
- **Language**: Python 3.12
- **Framework**: FastAPI
- **LLM**: Google Gemini Pro 1.5 (Vertex AI)
- **Monitoring**: Langfuse
- **Infrastructure**: Google Cloud Run

##  Economic Logic
マンデル＝フレミング・モデルの数式に基づき、利子率（$r$）と産出量（$Y$）の相関をシミュレーション。
$$IS: Y = C(Y-T) + I(r) + G + NX(e)$$
$$LM: \frac{M}{P} = L(r, Y)$$
これらの均衡点移動を JSON データとしてフロントに返し、正確なグラフ描画を支えています。
