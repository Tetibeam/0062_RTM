# Cython化 実行手順

## 現在の状態

✅ **完了:**
- Cython 3.2.1 のインストール
- Visual Studio Build Tools インストールガイドの作成
- バックアップ・削除スクリプトの作成

⏳ **次のステップ:**
- Visual Studio Build Tools のインストール

---

## ステップ1: Visual Studio Build Tools のインストール

詳細な手順は以下のファイルを参照してください:

📄 **[docs/visual_studio_build_tools_setup.md](file:///c:/dev/cases/portfolio/0041_Finannce_app_V002/docs/visual_studio_build_tools_setup.md)**

### 簡易手順:

1. **ダウンロード**: https://visualstudio.microsoft.com/downloads/
2. **Build Tools for Visual Studio 2022** をダウンロード
3. インストーラーで **「C++によるデスクトップ開発」** を選択
4. インストール実行 (約10-20分、6GB必要)

### インストール確認:

新しいPowerShellウィンドウを開いて実行:

```powershell
# Developer環境を読み込む
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"

# コンパイラの確認
cl.exe
```

---

## ステップ2: Cython化の実行

Visual Studio Build Toolsのインストール完了後:

```powershell
# Developer環境を読み込む(必要な場合)
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"

# Cython化を実行
python cythonize_batch_lib.py
```

**期待される結果:**
- `batch/lib/` 内に17個の `.pyd` ファイルが生成される
- 中間ファイル(`.c`, `build/`)は自動的にクリーンアップされる

---

## ステップ3: 動作確認

```powershell
# データベース初期化スクリプトで動作確認
python batch/init_db.py

# Webアプリケーションの起動確認
python app.py
```

---

## ステップ4: 元の.pyファイルの削除 (秘匿性の完全化)

動作確認が完了したら、元のPythonファイルを削除:

```powershell
# バックアップと削除を実行
python backup_and_remove_py.py
```

このスクリプトは以下を実行します:
1. `batch/lib/*.py` を `batch/lib_backup/` にバックアップ
2. `.pyd` ファイルの存在確認
3. 元の `.py` ファイルを削除

---

## トラブルシューティング

### cl.exeが見つからない

**Developer Command Prompt を使用:**
1. スタートメニューで「Developer Command Prompt for VS 2022」を検索
2. そのウィンドウ内で `python cythonize_batch_lib.py` を実行

**または、環境変数を読み込む:**
```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"
```

### コンパイルエラーが発生する

1. Visual Studio Build Toolsが正しくインストールされているか確認
2. 「C++によるデスクトップ開発」ワークロードが選択されているか確認
3. PCを再起動して環境変数を再読み込み

---

## 参考ファイル

- 📄 [cythonize_batch_lib.py](file:///c:/dev/cases/portfolio/0041_Finannce_app_V002/cythonize_batch_lib.py) - Cython化スクリプト
- 📄 [backup_and_remove_py.py](file:///c:/dev/cases/portfolio/0041_Finannce_app_V002/backup_and_remove_py.py) - バックアップ・削除スクリプト
- 📄 [docs/visual_studio_build_tools_setup.md](file:///c:/dev/cases/portfolio/0041_Finannce_app_V002/docs/visual_studio_build_tools_setup.md) - 詳細なインストールガイド
