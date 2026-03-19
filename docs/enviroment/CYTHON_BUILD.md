# batch/libディレクトリのCython化ガイド

このドキュメントでは、`batch/lib`ディレクトリ内のPythonファイルをCython化する方法を説明します。

## 目的

- **ソースコードの難読化**: `.py`ファイルをバイナリ(`.pyd`/`.so`)に変換し、ソースコードを読みにくくする
- **パフォーマンス向上**: Cythonコンパイルにより実行速度が向上する可能性がある

## 前提条件

### 1. Cythonのインストール

```powershell
pip install Cython
```

### 2. C++コンパイラのインストール

#### Windows
Visual Studio Build Toolsが必要です:
1. [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)をダウンロード
2. インストール時に「C++によるデスクトップ開発」を選択
3. または、既存のVisual Studioがあればそれを使用可能

#### Linux
```bash
sudo apt-get install build-essential  # Debian/Ubuntu
sudo yum install gcc gcc-c++          # RedHat/CentOS
```

#### macOS
```bash
xcode-select --install
```

## 使用方法

### 基本的な使い方

プロジェクトルートディレクトリで以下を実行:

```powershell
# 1. ドライラン(対象ファイルの確認のみ)
python cythonize_batch_lib.py --dry-run

# 2. 実際にコンパイル
python cythonize_batch_lib.py

# 3. 中間ファイルのクリーンアップ
python cythonize_batch_lib.py --clean
```

### オプション

| オプション | 説明 |
|-----------|------|
| `--dry-run` | 実際にはコンパイルせず、対象ファイルのみ表示 |
| `--clean` | ビルド中間ファイル(`.c`, `.cpp`, `build/`)を削除 |
| `--keep-c` | 中間Cファイルを保持(デバッグ用) |

## コンパイル後の動作

### ファイル構成

コンパイル後、`batch/lib`ディレクトリには以下のファイルが存在します:

```
batch/lib/
├── agg_asset_cleaning.py      # 元のPythonファイル
├── agg_asset_cleaning.pyd     # コンパイル済みバイナリ(Windows)
├── agg_asset_collection.py
├── agg_asset_collection.pyd
...
```

### インポートの優先順位

Pythonは以下の順序でモジュールを検索します:
1. `.pyd`/`.so`ファイル(コンパイル済みバイナリ)
2. `.py`ファイル(Pythonソース)

つまり、`.pyd`ファイルが存在する場合、元の`.py`ファイルは無視されます。

### デバッグ時の対応

デバッグが必要な場合は、`.pyd`/`.so`ファイルを一時的に削除してください:

```powershell
# Windowsの場合
Remove-Item batch\lib\*.pyd

# Linux/Macの場合
rm batch/lib/*.so
```

## トラブルシューティング

### エラー: "error: Microsoft Visual C++ 14.0 or greater is required"

**原因**: C++コンパイラがインストールされていない

**解決策**: Visual Studio Build Toolsをインストール
1. https://visualstudio.microsoft.com/downloads/ から「Build Tools for Visual Studio」をダウンロード
2. インストール時に「C++によるデスクトップ開発」を選択

### エラー: "ImportError: DLL load failed"

**原因**: コンパイル済みモジュールの依存関係の問題

**解決策**:
1. `.pyd`ファイルを削除して元の`.py`ファイルを使用
2. 仮想環境を再作成してから再コンパイル

### コンパイルは成功するが、インポート時にエラー

**原因**: Pythonバージョンの不一致

**解決策**: コンパイル時と実行時で同じPythonバージョンを使用していることを確認

```powershell
python --version  # バージョン確認
```

## 注意事項

1. **元のファイルは保持される**: `.py`ファイルは削除されず、`.pyd`/`.so`と共存します
2. **バージョン管理**: `.pyd`/`.so`ファイルは`.gitignore`に追加済みで、Gitには含まれません
3. **配布時**: 配布する際は、`.pyd`/`.so`ファイルのみを配布することでソースコードを保護できます
4. **パフォーマンス**: 全ての処理が高速化されるわけではありません。I/O処理が多い場合は効果が限定的です

## 元に戻す方法

コンパイル済みファイルを削除するだけで、元の`.py`ファイルが使用されます:

```powershell
# Windowsの場合
Remove-Item batch\lib\*.pyd
Remove-Item -Recurse build

# Linux/Macの場合
rm batch/lib/*.so
rm -rf build
```

または、クリーンアップスクリプトを使用:

```powershell
python cythonize_batch_lib.py --clean
```
