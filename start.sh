#!/bin/bash
echo "=========================================="
echo "  AI英会話アプリケーション セットアップ"
echo "=========================================="
echo

# 依存パッケージインストール
echo "依存パッケージをインストール中..."
pip install -r requirements.txt
echo

# サーバー起動
echo "サーバーを起動します..."
python server.py
