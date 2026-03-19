# Visual Studio Build Tools インストールガイド

Cythonでコンパイルを行うには、C/C++コンパイラが必要です。Windows環境では、Microsoft Visual Studio Build Toolsを使用します。

## インストール手順

### ステップ1: インストーラーのダウンロード

以下のリンクから**Build Tools for Visual Studio 2022**をダウンロードしてください:

🔗 **https://visualstudio.microsoft.com/downloads/**

ページ内の「**すべてのダウンロード**」セクションから、「**Build Tools for Visual Studio 2022**」を探してダウンロードします。

### ステップ2: インストーラーの実行

1. ダウンロードした`vs_BuildTools.exe`を実行
2. Visual Studio Installerが起動します

### ステップ3: ワークロードの選択

Visual Studio Installerで以下を選択:

✅ **「C++によるデスクトップ開発」** (Desktop development with C++)

このワークロードには以下が含まれます:
- MSVC v143 - VS 2022 C++ x64/x86 ビルドツール
- Windows SDK
- CMake ツール

### ステップ4: インストール

1. 右下の「**インストール**」ボタンをクリック
2. インストールには**10-20分程度**かかります
3. 必要なディスク容量: 約**6GB**

### ステップ5: インストールの確認

インストール完了後、**新しいPowerShellウィンドウ**を開いて以下のコマンドを実行:

```powershell
# Developer Command Promptを起動
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"

# コンパイラの確認
cl.exe
```

`cl.exe`が見つかれば、インストール成功です!

## 代替方法: Visual Studio Community Edition

既にVisual Studio Community Editionをインストールしている場合、または統合開発環境(IDE)も必要な場合は、こちらを使用できます:

🔗 **https://visualstudio.microsoft.com/vs/community/**

インストール時に「**C++によるデスクトップ開発**」ワークロードを選択してください。

## トラブルシューティング

### cl.exeが見つからない場合

1. **Developer Command Promptを使用**:
   - スタートメニューから「Developer Command Prompt for VS 2022」を検索して起動
   - そのウィンドウ内でPythonコマンドを実行

2. **環境変数の設定**:
   ```powershell
   # VS 2022 Build Toolsの環境変数を読み込む
   & "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"
   ```

3. **PCの再起動**:
   - インストール後、PCを再起動すると環境変数が正しく読み込まれることがあります

## 次のステップ

Visual Studio Build Toolsのインストールが完了したら、以下のコマンドでCython化を実行できます:

```powershell
# Cython化の実行
python cythonize_batch_lib.py
```
