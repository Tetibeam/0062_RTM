# React 化セットアップガイド

このドキュメントでは、React 化されたフロントエンドアプリケーションの起動方法を説明します。

## 前提条件

- **Node.js** (v18 以上推奨) および **npm** がインストールされていること
- Python の仮想環境が設定されていること

Node.js のインストール方法：

- [Node.js 公式サイト](https://nodejs.org/)からダウンロードしてインストール
- インストール後、以下のコマンドで確認:
  ```bash
  node --version
  npm --version
  ```

## セットアップ手順

### 1. バックエンド（Flask）の準備

バックエンドの依存関係をインストールします：

```bash
# プロジェクトルートで実行
pip install -r requirements.txt
```

Flask-CORS が新たに追加されています。

### 2. フロントエンド（React）の準備

フロントエンドの依存関係をインストールします：

```bash
# frontendディレクトリに移動
cd frontend

# 依存関係をインストール
npm install
```

## 開発サーバーの起動

React 化後は、**2 つのサーバー**を同時に起動する必要があります。

### ターミナル 1: バックエンド（Flask）

```bash
# プロジェクトルートで実行
python app.py
```

Flask サーバーが `http://localhost:5000` で起動します。

### ターミナル 2: フロントエンド（Vite）

```bash
# frontendディレクトリで実行
cd frontend
npm run dev
```

Vite 開発サーバーが `http://localhost:5173` で起動します。

## アクセス方法

ブラウザで以下の URL にアクセスします：

```
http://localhost:5173
```

## 動作確認

以下が正常に表示されれば成功です：

1. ✅ サイドバーに「💰 Finance App」とナビゲーションリンクが表示される
2. ✅ サイドバー下部に KPI ダッシュボード（Date, Fire Progress, Net Assets など）が表示される
3. ✅ メインエリアに 6 つのグラフ（FIRE Readiness, Savings Efficiency 等）が表示される
4. ✅ グラフタイトルをクリックするとフルスクリーン表示される
5. ✅ Back ボタンで通常表示に戻る

## トラブルシューティング

### CORS エラーが発生する場合

- バックエンド（Flask）が `http://localhost:5000` で起動していることを確認
- `app/__init__.py` で CORS 設定が正しく有効化されていることを確認

### グラフが表示されない場合

- ブラウザの開発者ツール（F12）で Network タブを確認
- `/api/Portfolio_Command_Center/graphs` へのリクエストが成功（200 OK）していることを確認
- データベースが初期化されていることを確認（`batch/init_master_db.py`を実行）

### npm install でエラーが出る場合

- Node.js のバージョンを確認（v18 以上推奨）
- npm のキャッシュをクリア: `npm cache clean --force`
- `frontend/node_modules` と `frontend/package-lock.json` を削除して再度`npm install`

## プロジェクト構造

```
0043_Fianance_app_V004/
├── app/                        # バックエンド（Flask）
│   ├── routes/                 # APIエンドポイント
│   ├── utils/                  # ユーティリティ
│   └── static/                 # 既存の静的ファイル（参照用に残す）
├── frontend/                   # フロントエンド（React）
│   ├── public/
│   │   └── icon/               # SVGアイコン
│   ├── src/
│   │   ├── components/         # 再利用可能なコンポーネント
│   │   │   ├── Sidebar.jsx
│   │   │   ├── KPIDashboard.jsx
│   │   │   └── GraphContainer.jsx
│   │   ├── pages/              # ページコンポーネント
│   │   │   └── PortfolioCommandCenter.jsx
│   │   ├── styles/
│   │   │   └── style.css
│   │   ├── App.jsx             # ルーティング設定
│   │   └── main.jsx            # エントリーポイント
│   ├── index.html
│   ├── vite.config.js          # Vite設定（プロキシ含む）
│   └── package.json
├── app.py                      # Flaskエントリーポイント
└── requirements.txt            # Python依存関係
```

## 次のステップ

今後のページ追加時は以下の手順で行います：

1. `frontend/src/pages/` に新しいページコンポーネントを作成
2. `frontend/src/App.jsx` にルート追加
3. `frontend/src/components/Sidebar.jsx` にナビゲーションリンク追加
4. 必要に応じてバックエンドに API エンドポイントを追加

## 参考

- [React 公式ドキュメント](https://react.dev/)
- [React Router 公式ドキュメント](https://reactrouter.com/)
- [Vite 公式ドキュメント](https://vitejs.dev/)
- [Plotly React](https://plotly.com/javascript/react/)
