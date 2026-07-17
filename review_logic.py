"""
復習モードの正誤判定ロジック（標準ライブラリのみ・純粋関数）
- LLMは使わず、正規化した文字列比較 + difflibの類似度で高速・オフライン判定する
- server.py の /api/review/answer から利用する
"""

import difflib
import re

# 「惜しい」判定のしきい値（正規化後テキストの類似度）
CLOSE_RATIO_THRESHOLD = 0.85


def normalize_review_text(text: str) -> str:
    """
    比較用にテキストを正規化する。
    - 小文字化
    - 前後空白の除去
    - 連続空白の1個への圧縮
    - 末尾の句読点（. , ! ? ; :）の無視
    """
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    # 末尾の句読点類（連続していてもまとめて）を除去し、残った末尾空白も落とす
    t = re.sub(r"[.,!?;:]+$", "", t).rstrip()
    return t


def judge_review_answer(answer_text: str, corrected: str) -> dict:
    """
    解答テキストと正答（corrected）を正規化して比較する。
    - 完全一致: correct=True
    - 不一致でも類似度 >= CLOSE_RATIO_THRESHOLD なら close=True（惜しい）
    戻り値: {"correct": bool, "close": bool, "ratio": float}
    """
    answer = normalize_review_text(answer_text)
    target = normalize_review_text(corrected)

    # どちらかが空なら判定不能（不正解扱い）
    if not answer or not target:
        return {"correct": False, "close": False, "ratio": 0.0}

    if answer == target:
        return {"correct": True, "close": False, "ratio": 1.0}

    ratio = difflib.SequenceMatcher(None, answer, target).ratio()
    return {
        "correct": False,
        "close": ratio >= CLOSE_RATIO_THRESHOLD,
        "ratio": round(ratio, 3),
    }
