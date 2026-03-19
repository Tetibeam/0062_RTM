# Docker Desktop起動問題のトラブルシューティング

## 問題の概要

`docker info`コマンドで以下のエラーが発生しています:
```
ERROR: Error response from daemon: Docker Desktop is unable to start
```

これはDocker Desktopアプリケーション自体が正常に起動できていないことを示しています。

---

## トラブルシューティング手順

### 1. Docker Desktopの再起動

#### 方法A: タスクマネージャーから強制終了
1. `Ctrl + Shift + Esc`でタスクマネージャーを開く
2. 「Docker Desktop」プロセスを探す
3. 右クリック → 「タスクの終了」
4. スタートメニューから「Docker Desktop」を再起動

#### 方法B: PowerShellから再起動
```powershell
# Docker Desktopを停止
Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue

# Docker Desktopを起動
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

### 2. WSL2の確認（Windows Subsystem for Linux）

Docker DesktopはWSL2を使用しています。WSL2が正常に動作しているか確認します。

```powershell
# WSLのバージョン確認
wsl --version

# WSLの状態確認
wsl --list --verbose

# WSLを再起動
wsl --shutdown
```

### 3. Hyper-Vの確認

Docker DesktopにはHyper-Vまたは仮想化機能が必要です。

```powershell
# 管理者権限のPowerShellで実行
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V
```

有効になっていない場合:
```powershell
# 管理者権限のPowerShellで実行
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
```

### 4. Docker Desktopのログ確認

Docker Desktopのログを確認して詳細なエラーを調査します。

ログファイルの場所:
```
%APPDATA%\Docker\log.txt
```

PowerShellで確認:
```powershell
Get-Content "$env:APPDATA\Docker\log.txt" -Tail 50
```

### 5. Docker Desktopの再インストール

上記の方法で解決しない場合、Docker Desktopを再インストールします。

1. **アンインストール**
   - 設定 → アプリ → Docker Desktop → アンインストール

2. **クリーンアップ**
   ```powershell
   # Docker関連ディレクトリを削除
   Remove-Item -Path "$env:APPDATA\Docker" -Recurse -Force -ErrorAction SilentlyContinue
   Remove-Item -Path "$env:LOCALAPPDATA\Docker" -Recurse -Force -ErrorAction SilentlyContinue
   ```

3. **再インストール**
   - https://www.docker.com/products/docker-desktop からダウンロード
   - インストーラーを実行

---

## 代替案: PostgreSQLのネイティブインストール

Docker環境の問題が解決しない場合、PostgreSQLを直接Windowsにインストールすることもできます。

### PostgreSQL for Windowsのインストール

1. **ダウンロード**
   - https://www.postgresql.org/download/windows/
   - EDB Installerを推奨

2. **インストール設定**
   - ポート: 5432
   - ユーザー名: `finance_user`
   - パスワード: `finance_password`
   - データベース名: `finance_db`

3. **データベースの作成**
   ```sql
   CREATE DATABASE finance_db;
   CREATE USER finance_user WITH PASSWORD 'finance_password';
   GRANT ALL PRIVILEGES ON DATABASE finance_db TO finance_user;
   ```

4. **`.env`ファイルの設定**
   ```bash
   DB_TYPE=postgresql
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=finance_db
   DB_USER=finance_user
   DB_PASSWORD=finance_password
   ```

---

## 現在の状態

現在、プロジェクトはSQLiteモードで動作しています（`.env`で`DB_TYPE=sqlite`に設定済み）。

### SQLiteモードでの動作確認

データベース抽象化レイヤーは正常に動作しています:
```
Database Type: sqlite
Database Path: C:\dev\cases\portfolio\0042_Finannce_app_V003\database\finance.db
```

### PostgreSQLへの切り替え

Docker DesktopまたはネイティブPostgreSQLが正常に動作したら、`.env`ファイルを編集:
```bash
DB_TYPE=postgresql
```

---

## 推奨される次のステップ

1. **まずSQLite環境で検証を完了する**
   - データベース初期化テスト
   - アプリケーション動作確認
   - データ取得・更新機能のテスト

2. **Docker Desktop問題の解決**
   - 上記のトラブルシューティング手順を試す
   - または、PostgreSQLのネイティブインストールを検討

3. **PostgreSQL環境での検証**
   - `.env`を`postgresql`に変更
   - データベース初期化
   - 動作確認

---

## サポート情報

### Docker Desktop公式トラブルシューティング
- https://docs.docker.com/desktop/troubleshoot/overview/

### WSL2のトラブルシューティング
- https://learn.microsoft.com/ja-jp/windows/wsl/troubleshooting

### PostgreSQL公式ドキュメント
- https://www.postgresql.org/docs/
