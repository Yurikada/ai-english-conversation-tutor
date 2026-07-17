"""
AI英会話アプリケーション - SQLite永続化レイヤー
sqlite3標準ライブラリのみ使用（外部依存なし）
FastAPIのasyncコンテキストから安全に呼べるよう、毎回接続方式を採用
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    # 通常はconfig.pyのBASE_DIRを使用
    from config import BASE_DIR
except ImportError:
    # 単体テスト等でconfigが無い場合は自身のディレクトリを使用
    BASE_DIR = Path(__file__).resolve().parent

# エラー種別の自動タグ付け（ルールベース・LLM呼び出しなし）
from error_tagger import tag_correction

# DBファイルのパス（テスト時は db.DB_PATH を差し替え可能）
DB_PATH = BASE_DIR / "tutor.db"


def _now() -> str:
    """現在時刻をISO 8601形式（UTC）で返す"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_connection():
    """毎回新規接続するcontextmanager（スレッド安全・自動commit/close）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """テーブルを初期化する（存在しなければ作成）"""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                created_at TEXT NOT NULL,
                user_text TEXT NOT NULL,
                reply TEXT NOT NULL,
                natural_expression TEXT,
                encouragement TEXT,
                pronunciation_score INTEGER
            );

            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL REFERENCES turns(id),
                original TEXT,
                corrected TEXT,
                explanation TEXT,
                error_type TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_corrections_turn ON corrections(turn_id);
        """)

        # 既存DB互換: error_type カラムが無い旧テーブルにはALTERで追加する
        # （既に存在する場合は duplicate column エラーになるので握りつぶす）
        try:
            conn.execute("ALTER TABLE corrections ADD COLUMN error_type TEXT")
        except sqlite3.OperationalError:
            pass

        # 既存DB互換: 復習モード（簡易SRS）用のカラムをALTERで追加する
        # 既に存在する場合は duplicate column エラーになるので個別に握りつぶす
        for column_def in (
            "review_count INTEGER DEFAULT 0",      # 復習した回数
            "correct_streak INTEGER DEFAULT 0",    # 連続正解数（不正解でリセット）
            "next_review_at TEXT",                 # 次回復習予定時刻（NULL=未復習）
            "last_reviewed_at TEXT",               # 最後に復習した時刻
        ):
            try:
                conn.execute(f"ALTER TABLE corrections ADD COLUMN {column_def}")
            except sqlite3.OperationalError:
                pass


def save_turn(session_id: str, user_text: str, ai_response: dict, score: int) -> int:
    """
    1ターン分の会話をDBに保存する。
    - セッションが未登録なら作成
    - turnsに本体を保存し、correctionsに添削リストを保存
    - 保存したturnのIDを返す
    """
    now = _now()
    with get_connection() as conn:
        # セッションが無ければ登録
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, now),
        )

        cur = conn.execute(
            """
            INSERT INTO turns
                (session_id, created_at, user_text, reply,
                 natural_expression, encouragement, pronunciation_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                now,
                user_text,
                ai_response.get("reply", ""),
                ai_response.get("natural_expression"),
                ai_response.get("encouragement", ""),
                score,
            ),
        )
        turn_id = cur.lastrowid

        # 添削リストを保存（不正な要素はスキップ）
        for correction in ai_response.get("corrections", []) or []:
            if not isinstance(correction, dict):
                continue
            original = correction.get("original", "")
            corrected = correction.get("corrected", "")
            explanation = correction.get("explanation", "")
            # ルールベースでエラー種別を自動タグ付け（LLM呼び出しなし）
            error_type = tag_correction(original, corrected, explanation)
            conn.execute(
                """
                INSERT INTO corrections
                    (turn_id, original, corrected, explanation, error_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (turn_id, original, corrected, explanation, error_type),
            )

        return turn_id


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    """
    セッションの会話履歴をLLMに渡すmessages形式で返す。
    limitは最大メッセージ数（1ターン = user + assistant の2メッセージ）
    """
    turn_limit = max(1, limit // 2)
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT user_text, reply FROM turns
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, turn_limit),
        ).fetchall()

    messages: list[dict] = []
    # 新しい順で取得しているので古い順に戻す
    for row in reversed(rows):
        messages.append({"role": "user", "content": row["user_text"]})
        messages.append({"role": "assistant", "content": row["reply"]})
    return messages


def get_session_summary_data(session_id: str) -> dict | None:
    """
    セッションサマリ（今日の学びレポート）用のデータを返す。
    - turns: user_text / reply / pronunciation_score / natural_expression（古い順）
    - corrections: original / corrected / explanation / error_type（古い順）
    そのセッションにターンが1件も無ければ None を返す。
    """
    with get_connection() as conn:
        turn_rows = conn.execute(
            """
            SELECT user_text, reply, pronunciation_score, natural_expression
            FROM turns
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

        # ターンが無いセッション（未使用 or 存在しないID）はサマリ対象外
        if not turn_rows:
            return None

        correction_rows = conn.execute(
            """
            SELECT c.original, c.corrected, c.explanation, c.error_type
            FROM corrections c
            JOIN turns t ON t.id = c.turn_id
            WHERE t.session_id = ?
            ORDER BY c.id ASC
            """,
            (session_id,),
        ).fetchall()

    return {
        "turns": [
            {
                "user_text": row["user_text"],
                "reply": row["reply"],
                "pronunciation_score": row["pronunciation_score"],
                "natural_expression": row["natural_expression"],
            }
            for row in turn_rows
        ],
        "corrections": [
            {
                "original": row["original"],
                "corrected": row["corrected"],
                "explanation": row["explanation"],
                "error_type": row["error_type"],
            }
            for row in correction_rows
        ],
    }


def get_recent_sessions(limit: int = 20) -> list[dict]:
    """最近のセッション一覧（ターン数・平均スコア・最終更新付き）を返す"""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.created_at,
                COUNT(t.id) AS turn_count,
                ROUND(AVG(t.pronunciation_score)) AS avg_score,
                MAX(t.created_at) AS last_active
            FROM sessions s
            LEFT JOIN turns t ON t.session_id = s.id
            GROUP BY s.id
            ORDER BY COALESCE(MAX(t.created_at), s.created_at) DESC,
                     COALESCE(MAX(t.id), 0) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "turn_count": row["turn_count"],
            "avg_score": row["avg_score"],
            "last_active": row["last_active"],
        }
        for row in rows
    ]


def get_all_corrections(limit: int = 100) -> list[dict]:
    """全セッションの添削履歴を新しい順で返す（弱点ダッシュボード用）"""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.original,
                c.corrected,
                c.explanation,
                c.error_type,
                t.session_id,
                t.user_text,
                t.created_at
            FROM corrections c
            JOIN turns t ON t.id = c.turn_id
            ORDER BY c.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "original": row["original"],
            "corrected": row["corrected"],
            "explanation": row["explanation"],
            "error_type": row["error_type"],
            "session_id": row["session_id"],
            "user_text": row["user_text"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_error_stats(example_limit: int = 3) -> list[dict]:
    """
    エラー種別ごとの件数集計と直近の例文を返す（弱点ダッシュボード用）。
    error_typeが未設定（旧データ）の行は "other" として集計する。
    戻り値: [{error_type, count, examples: [{original, corrected, explanation}]}] 件数降順
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                COALESCE(NULLIF(error_type, ''), 'other') AS error_type,
                COUNT(*) AS count
            FROM corrections
            GROUP BY 1
            ORDER BY count DESC, error_type ASC
            """
        ).fetchall()

        stats = []
        for row in rows:
            # 各種別の直近の添削例を数件取得する
            examples = conn.execute(
                """
                SELECT original, corrected, explanation
                FROM corrections
                WHERE COALESCE(NULLIF(error_type, ''), 'other') = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (row["error_type"], example_limit),
            ).fetchall()
            stats.append({
                "error_type": row["error_type"],
                "count": row["count"],
                "examples": [
                    {
                        "original": ex["original"],
                        "corrected": ex["corrected"],
                        "explanation": ex["explanation"],
                    }
                    for ex in examples
                ],
            })

    return stats


def get_dashboard_stats(score_limit: int = 20) -> dict:
    """
    ダッシュボード用の全体統計を返す。
    - total_turns / total_corrections / avg_score
    - recent_scores: 直近の発音スコア推移（古い順）
    """
    with get_connection() as conn:
        total_turns = conn.execute(
            "SELECT COUNT(*) AS n FROM turns"
        ).fetchone()["n"]
        total_corrections = conn.execute(
            "SELECT COUNT(*) AS n FROM corrections"
        ).fetchone()["n"]
        avg_row = conn.execute(
            "SELECT ROUND(AVG(pronunciation_score), 1) AS avg FROM turns "
            "WHERE pronunciation_score IS NOT NULL"
        ).fetchone()
        # 直近スコアを新しい順で取り、表示用に古い順へ戻す
        score_rows = conn.execute(
            """
            SELECT pronunciation_score AS score, created_at
            FROM turns
            WHERE pronunciation_score IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (score_limit,),
        ).fetchall()

    return {
        "total_turns": total_turns,
        "total_corrections": total_corrections,
        "avg_score": avg_row["avg"],
        "recent_scores": [row["score"] for row in reversed(score_rows)],
    }


# ============================================================
# 復習モード（簡易SRS: Spaced Repetition System）
# ============================================================
# 連続正解数に応じた復習間隔（日数）。
# 1回目正解=1日後、2回目=3日後、3回目=7日後、4回目=14日後、5回目以降=30日後
SRS_INTERVALS_DAYS = [1, 3, 7, 14, 30]


def get_due_reviews(limit: int = 10) -> list[dict]:
    """
    復習対象のcorrectionsを返す。
    - next_review_at がNULL（未復習）または現在時刻以前のものが対象
    - よく間違えるerror_type（全correctionsでの頻度）の高い順 → 古い順（id昇順）
    - correctedが空のものは出題できないため除外
    """
    now = _now()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.original,
                c.corrected,
                c.explanation,
                c.error_type,
                c.review_count,
                t.user_text
            FROM corrections c
            JOIN turns t ON t.id = c.turn_id
            JOIN (
                -- error_typeごとの出現頻度（未設定は 'other' に正規化して集計）
                SELECT COALESCE(NULLIF(error_type, ''), 'other') AS et,
                       COUNT(*) AS freq
                FROM corrections
                GROUP BY 1
            ) f ON f.et = COALESCE(NULLIF(c.error_type, ''), 'other')
            WHERE (c.next_review_at IS NULL OR c.next_review_at <= ?)
              AND c.corrected IS NOT NULL AND TRIM(c.corrected) != ''
            ORDER BY f.freq DESC, c.id ASC
            LIMIT ?
            """,
            (now, limit),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "original": row["original"],
            "corrected": row["corrected"],
            "explanation": row["explanation"],
            "error_type": row["error_type"],
            "review_count": row["review_count"] or 0,
            "user_text": row["user_text"],
        }
        for row in rows
    ]


def get_correction(correction_id: int) -> dict | None:
    """correctionを1件取得する（復習の正誤判定用）。存在しなければNone"""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, original, corrected, explanation, error_type,
                   review_count, correct_streak, next_review_at, last_reviewed_at
            FROM corrections
            WHERE id = ?
            """,
            (correction_id,),
        ).fetchone()
    return dict(row) if row else None


def record_review_result(correction_id: int, correct: bool) -> dict | None:
    """
    復習の正誤結果で簡易SRS状態を更新する。
    - 正解: correct_streak+1。間隔は更新前のstreakで SRS_INTERVALS_DAYS を引く
      （1回目正解=1日後、2回目=3日後、…、5回目以降=30日後）
    - 不正解: correct_streak=0、1日後に再出題
    - 共通: review_count+1、last_reviewed_at更新
    更新後の状態dictを返す。対象が存在しなければNone。
    """
    now_dt = datetime.now(timezone.utc)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT review_count, correct_streak FROM corrections WHERE id = ?",
            (correction_id,),
        ).fetchone()
        if row is None:
            return None

        old_streak = row["correct_streak"] or 0
        if correct:
            new_streak = old_streak + 1
            # 更新前streakをインデックスに（上限は30日で頭打ち）
            interval_days = SRS_INTERVALS_DAYS[min(old_streak, len(SRS_INTERVALS_DAYS) - 1)]
        else:
            new_streak = 0
            interval_days = 1  # 不正解は翌日に再挑戦

        next_review_at = (now_dt + timedelta(days=interval_days)).isoformat(timespec="seconds")
        last_reviewed_at = now_dt.isoformat(timespec="seconds")
        review_count = (row["review_count"] or 0) + 1

        conn.execute(
            """
            UPDATE corrections
            SET review_count = ?, correct_streak = ?,
                next_review_at = ?, last_reviewed_at = ?
            WHERE id = ?
            """,
            (review_count, new_streak, next_review_at, last_reviewed_at, correction_id),
        )

    return {
        "correction_id": correction_id,
        "review_count": review_count,
        "correct_streak": new_streak,
        "interval_days": interval_days,
        "next_review_at": next_review_at,
        "last_reviewed_at": last_reviewed_at,
    }
