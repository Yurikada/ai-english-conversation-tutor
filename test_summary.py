"""
セッションサマリ（今日の学びレポート）のテスト
- db.get_session_summary_data の単体テスト
- GET /api/summary/{session_id} のFastAPI TestClientテスト（LLMはモック）
実行方法: python test_summary.py  （pytest不要・標準のassertのみ）
"""

import json
import os
import tempfile
from pathlib import Path

# テスト用の一時DBを先に決めてから db を import する
TMP_DIR = tempfile.mkdtemp(prefix="summary_test_")
TEST_DB_PATH = Path(TMP_DIR) / "test_tutor.db"

import db
db.DB_PATH = TEST_DB_PATH

import server
from fastapi.testclient import TestClient

PASSED = []


def check(name: str, condition: bool, detail: str = ""):
    """簡易アサーション（失敗時は即例外で停止）"""
    assert condition, f"FAILED: {name} {detail}"
    PASSED.append(name)
    print(f"  ok - {name}")


def make_session_data():
    """テスト用のセッションデータを2ターン分投入する"""
    db.init_db()
    session_id = "sess_test_1"
    db.save_turn(
        session_id,
        "I go to school yesterday",
        {
            "reply": "What did you do at school?",
            "corrections": [
                {"original": "I go to school yesterday",
                 "corrected": "I went to school yesterday",
                 "explanation": "Use past tense 'went' with 'yesterday'"},
            ],
            "natural_expression": "I went to school yesterday.",
            "encouragement": "Nice try!",
        },
        72,
    )
    db.save_turn(
        session_id,
        "I had a great time at the party",
        {
            "reply": "That sounds fun! What was the best part?",
            "corrections": [],
            "natural_expression": None,
            "encouragement": "Great!",
        },
        90,
    )
    return session_id


# ============================================================
# 1. db.get_session_summary_data の単体テスト
# ============================================================
def test_db_get_session_summary_data():
    print("[1] db.get_session_summary_data")
    session_id = make_session_data()

    data = db.get_session_summary_data(session_id)
    check("turns数", data is not None and len(data["turns"]) == 2)
    check("turnsが古い順", data["turns"][0]["user_text"] == "I go to school yesterday")
    check("turnにreplyが含まれる", data["turns"][0]["reply"] == "What did you do at school?")
    check("turnにスコアが含まれる",
          [t["pronunciation_score"] for t in data["turns"]] == [72, 90])
    check("natural_expressionが含まれる",
          data["turns"][0]["natural_expression"] == "I went to school yesterday.")
    check("corrections数", len(data["corrections"]) == 1)
    c = data["corrections"][0]
    check("correctionの中身",
          c["original"] == "I go to school yesterday"
          and c["corrected"] == "I went to school yesterday"
          and c["explanation"].startswith("Use past tense"))
    # error_tagger により時制エラーとタグ付けされるはず
    check("error_typeが付与される", c["error_type"] == "tense", f"got {c['error_type']}")

    # turnが無いセッションは None
    check("存在しないセッションはNone", db.get_session_summary_data("no_such_session") is None)


# ============================================================
# 2. parse_summary_llm_json の単体テスト
# ============================================================
def test_parse_summary_llm_json():
    print("[2] server.parse_summary_llm_json")

    # コードブロック + 前置きテキスト入りでも抽出できる
    raw = ('Here is your report!\n```json\n'
           '{"highlights": ["Good tense usage", "Nice vocabulary"], '
           '"focus_areas": ["Articles", "Prepositions"], '
           '"phrase_to_remember": "I went to school yesterday.", '
           '"message": "Keep it up!"}\n```')
    fb = server.parse_summary_llm_json(raw)
    check("コードブロック除去して抽出", fb["highlights"] == ["Good tense usage", "Nice vocabulary"])
    check("focus_areas抽出", fb["focus_areas"] == ["Articles", "Prepositions"])
    check("phrase抽出", fb["phrase_to_remember"] == "I went to school yesterday.")
    check("message抽出", fb["message"] == "Keep it up!")

    # 壊れた応答はフォールバック（空のhighlights等）
    fb2 = server.parse_summary_llm_json("sorry, I can't do that {broken json")
    check("壊れたJSONはフォールバック",
          fb2 == {"highlights": [], "focus_areas": [],
                  "phrase_to_remember": None, "message": ""})

    # 空文字もフォールバック
    fb3 = server.parse_summary_llm_json("")
    check("空応答はフォールバック", fb3["highlights"] == [] and fb3["phrase_to_remember"] is None)

    # 型が不正なフィールドは正規化される（highlightsがstr、phraseが数値）
    fb4 = server.parse_summary_llm_json(
        '{"highlights": "only one", "focus_areas": [1, "two"], '
        '"phrase_to_remember": 123, "message": null}')
    check("不正型の正規化",
          fb4["highlights"] == ["only one"]
          and fb4["focus_areas"] == ["1", "two"]
          and fb4["phrase_to_remember"] is None
          and fb4["message"] == "")


# ============================================================
# 3. GET /api/summary/{session_id} のAPIテスト（LLMモック）
# ============================================================
def test_api_summary():
    print("[3] GET /api/summary/{session_id}")
    session_id = "sess_test_1"  # test_db で投入済み
    original_get_llm_response = server.get_llm_response

    client = TestClient(server.app)

    # ---- ケース1: LLM成功（コードブロック付きJSONを返す） ----
    async def mock_llm_ok(messages):
        # プロンプトに会話と添削が含まれていることも確認
        joined = json.dumps(messages, ensure_ascii=False)
        assert "I go to school yesterday" in joined
        assert "I went to school yesterday" in joined
        return ('```json\n{"highlights": ["h1", "h2"], "focus_areas": ["f1", "f2"], '
                '"phrase_to_remember": "I went to school yesterday.", '
                '"message": "Great job today!"}\n```')

    server.get_llm_response = mock_llm_ok
    try:
        resp = client.get(f"/api/summary/{session_id}")
        check("成功時200", resp.status_code == 200)
        body = resp.json()
        st = body["stats"]
        check("turn_count", st["turn_count"] == 2)
        check("avg_score", st["avg_score"] == 81.0, f"got {st['avg_score']}")
        check("max/min_score", st["max_score"] == 90 and st["min_score"] == 72)
        check("correction_count", st["correction_count"] == 1)
        check("エラー種別内訳", st["error_types"] == [{"error_type": "tense", "count": 1}])
        check("natural_expressions",
              st["natural_expressions"] == ["I went to school yesterday."])
        fb = body["feedback"]
        check("LLM講評が返る",
              fb["highlights"] == ["h1", "h2"] and fb["focus_areas"] == ["f1", "f2"]
              and fb["phrase_to_remember"] == "I went to school yesterday."
              and fb["message"] == "Great job today!")

        # ---- ケース2: LLM失敗 → 統計のみで200、feedbackはnull ----
        async def mock_llm_fail(messages):
            raise RuntimeError("LLM connection error (simulated)")

        server.get_llm_response = mock_llm_fail
        resp2 = client.get(f"/api/summary/{session_id}")
        check("LLM失敗でも200", resp2.status_code == 200)
        body2 = resp2.json()
        check("LLM失敗時feedbackはnull", body2["feedback"] is None)
        check("LLM失敗でも統計は返る", body2["stats"]["turn_count"] == 2)

        # ---- ケース3: turnが無いセッションは404（LLMは呼ばれない） ----
        llm_called = []

        async def mock_llm_spy(messages):
            llm_called.append(True)
            return "{}"

        server.get_llm_response = mock_llm_spy
        resp3 = client.get("/api/summary/no_such_session")
        check("未知セッションは404", resp3.status_code == 404)
        check("404時はLLM未呼び出し", llm_called == [])

        # ---- ケース4: LLMが壊れたJSONを返す → フォールバック講評で200 ----
        async def mock_llm_garbage(messages):
            return "I am sorry, I cannot produce JSON today."

        server.get_llm_response = mock_llm_garbage
        resp4 = client.get(f"/api/summary/{session_id}")
        check("壊れたJSONでも200", resp4.status_code == 200)
        fb4 = resp4.json()["feedback"]
        check("壊れたJSONは空フォールバック講評",
              fb4 is not None and fb4["highlights"] == [] and fb4["message"] == "")
    finally:
        server.get_llm_response = original_get_llm_response


if __name__ == "__main__":
    test_db_get_session_summary_data()
    test_parse_summary_llm_json()
    test_api_summary()
    print(f"\nAll {len(PASSED)} checks passed.")
