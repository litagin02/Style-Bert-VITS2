#!/bin/bash
echo "===================================="
echo "Style-Bert-VITS2 環境構築スクリプト"
echo "===================================="
echo

# Python 3.10の確認
if ! python3 --version | grep -q "3.10"; then
    echo "[ERROR] Python 3.10.xが必要です。"
    echo "pyenvまたはaptでPython 3.10をインストールしてください。"
    exit 1
fi

echo "[1/5] 既存の仮想環境を削除中..."
if [ -d "venv" ]; then
    rm -rf venv
fi

echo "[2/5] 新しい仮想環境を作成中..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] 仮想環境の作成に失敗しました"
    exit 1
fi

echo "[3/5] 仮想環境を有効化中..."
source venv/bin/activate

echo "[4/5] pipをアップグレード中..."
python -m pip install --upgrade pip

echo "[5/5] 依存関係をインストール中..."
pip install -r requirements.txt

echo
echo "===================================="
echo "セットアップ完了！"
echo "===================================="
echo
echo "仮想環境を有効化するには以下を実行してください:"
echo "  source venv/bin/activate"
echo
echo "アプリケーションを起動するには:"
echo "  python app.py"
echo