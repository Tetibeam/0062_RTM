# PostgreSQL環境セットアップガイド

このドキュメントでは、Docker Composeを使用してPostgreSQL環境をセットアップする手順を説明します。

## 前提条件

- Docker Desktopがインストールされていること
- Gitがインストールされていること

## セットアップ手順

### 1. Docker Desktopの確認

Docker Desktopが起動していることを確認してください。

```bash
docker --version
docker-compose --version
```

### 2. PostgreSQL環境の起動

プロジェクトルートで以下のコマンドを実行します:

```bash
# PostgreSQLコンテナを起動（バックグラウンド）
docker-compose up -d

# ログを確認
docker-compose logs -f postgres
```

### 3. 接続確認

PostgreSQLが正常に起動したことを確認します:

```bash
# ヘルスチェック
docker-compose ps

# PostgreSQLに接続
docker-compose exec postgres psql -U finance_user -d finance_db
```

psqlプロンプトが表示されたら成功です。`\q`で終了できます。

### 4. 環境変数の設定

`.env`ファイルが作成されていることを確認してください。必要に応じて編集します:

```bash
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finance_db
DB_USER=finance_user
DB_PASSWORD=finance_password
```

### 5. Pythonパッケージのインストール

仮想環境を有効化し、必要なパッケージをインストールします:

```bash
# 仮想環境の作成（初回のみ）
python -m venv .venv

# 仮想環境の有効化
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# パッケージのインストール
pip install -r requirements.txt
```

## PostgreSQL環境の管理

### コンテナの起動・停止

```bash
# 起動
docker-compose up -d

# 停止
docker-compose down

# 停止してデータも削除
docker-compose down -v
```

### データベースの初期化

```bash
# 初期化スクリプトを実行
python batch/init_db.py
```

### ログの確認

```bash
# リアルタイムでログを表示
docker-compose logs -f postgres

# 最新100行を表示
docker-compose logs --tail=100 postgres
```

## トラブルシューティング

### ポート5432が既に使用されている

別のPostgreSQLが起動している可能性があります。`docker-compose.yml`のポート設定を変更してください:

```yaml
ports:
  - "5433:5432"  # ホスト側を5433に変更
```

`.env`ファイルも更新:
```
DB_PORT=5433
```

### コンテナが起動しない

```bash
# コンテナの状態を確認
docker-compose ps

# ログを確認
docker-compose logs postgres

# コンテナを再作成
docker-compose down
docker-compose up -d --force-recreate
```

### データベースに接続できない

1. PostgreSQLコンテナが起動しているか確認
2. `.env`ファイルの設定が正しいか確認
3. ファイアウォールの設定を確認

## SQLiteとの切り替え

開発中にSQLiteを使用したい場合は、`.env`ファイルを編集:

```bash
DB_TYPE=sqlite
SQLITE_DB_PATH=./database/finance.db
```

## 参考情報

- PostgreSQL公式ドキュメント: https://www.postgresql.org/docs/
- Docker Compose公式ドキュメント: https://docs.docker.com/compose/
