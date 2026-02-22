# 1. 安定した Python 3.11 を使用
FROM python:3.11-slim

# 2. ログをリアルタイムで表示する設定
ENV PYTHONUNBUFFERED=1

# 3. 作業ディレクトリの設定
WORKDIR /app

# 4. 必要なライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. ソースコードをすべてコピー
COPY . .

# 6. アプリを起動（ポート8080を明示）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
