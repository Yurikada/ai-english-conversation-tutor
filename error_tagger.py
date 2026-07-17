"""
エラー種別の自動タグ付けモジュール（ルールベース）
LLMの追加呼び出しは行わず、添削のexplanation内キーワードと
original/correctedの簡易diffヒューリスティックだけで分類する。

カテゴリ:
    tense        時制
    article      冠詞 (a/an/the)
    preposition  前置詞
    word_order   語順
    subject_verb 主語と動詞の一致
    plural       単数・複数
    vocabulary   語彙選択
    spelling     スペル
    other        その他

判定の流れ（最初にマッチしたものを返す）:
    1. explanationのキーワード判定（優先順位: word_order → subject_verb →
       tense → article → preposition → plural → spelling → vocabulary）
       ※「third-person singular」等の複合表現があるため、より特定的な
         カテゴリを先に判定する順序にしている
    2. original/correctedのdiffヒューリスティック
    3. どれにも該当しなければ "other"
"""

import difflib
import re

# 有効なカテゴリ一覧（外部からの参照用）
ERROR_TYPES = [
    "tense", "article", "preposition", "word_order",
    "subject_verb", "plural", "vocabulary", "spelling", "other",
]

# ------------------------------------------------------------
# 1. explanationキーワード判定（優先順位順・正規表現）
# ------------------------------------------------------------
_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("word_order", [
        r"word order", r"order of (?:the )?words?", r"語順", r"並び順",
    ]),
    ("subject_verb", [
        r"agreement", r"subject[- ]verb", r"third[- ]person", r"3rd[- ]person",
        r"主語と動詞", r"三単現", r"三人称単数",
    ]),
    ("tense", [
        r"tense", r"past (?:form|simple|participle)", r"present perfect",
        r"past continuous", r"future", r"時制", r"過去形", r"現在完了",
        r"過去分詞", r"未来形",
    ]),
    ("article", [
        r"article", r"冠詞", r"\ba/an\b", r"['\"]a['\"]", r"['\"]an['\"]",
        r"['\"]the['\"]", r"\bdefinite\b", r"\bindefinite\b",
    ]),
    ("preposition", [
        r"preposition", r"前置詞",
    ]),
    ("plural", [
        r"plural", r"singular", r"uncountable", r"countable",
        r"複数形", r"単数形", r"単複", r"可算", r"不可算",
    ]),
    ("spelling", [
        r"spell", r"typo", r"misspel", r"スペル", r"綴り",
    ]),
    ("vocabulary", [
        r"word choice", r"vocabulary", r"wrong word", r"better word",
        r"correct word", r"more natural word", r"語彙", r"言葉の選", r"単語の選",
    ]),
]

# コンパイル済みパターン（カテゴリごとにOR結合）
_COMPILED_RULES = [
    (etype, re.compile("|".join(patterns), re.IGNORECASE))
    for etype, patterns in _KEYWORD_RULES
]

# ------------------------------------------------------------
# 2. diffヒューリスティック用の語彙データ
# ------------------------------------------------------------
_ARTICLES = {"a", "an", "the"}

_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
    "into", "onto", "during", "since", "until", "over", "under",
    "between", "through", "before", "after", "against",
}

# 主語動詞一致でよく入れ替わるペア（順不同）
_SV_PAIRS = {
    frozenset(p) for p in [
        ("is", "are"), ("was", "were"), ("has", "have"),
        ("does", "do"), ("doesn't", "don't"), ("am", "are"), ("am", "is"),
    ]
}

# 不規則動詞の 原形→過去形 ペア（順不同で照合）
_IRREGULAR_PAST_PAIRS = {
    frozenset(p) for p in [
        ("go", "went"), ("eat", "ate"), ("see", "saw"), ("do", "did"),
        ("have", "had"), ("come", "came"), ("get", "got"), ("take", "took"),
        ("make", "made"), ("say", "said"), ("buy", "bought"),
        ("think", "thought"), ("teach", "taught"), ("catch", "caught"),
        ("run", "ran"), ("write", "wrote"), ("speak", "spoke"),
        ("meet", "met"), ("find", "found"), ("give", "gave"),
        ("know", "knew"), ("tell", "told"), ("feel", "felt"),
        ("leave", "left"), ("begin", "began"), ("drink", "drank"),
        ("swim", "swam"), ("sing", "sang"), ("sleep", "slept"),
    ]
}

# be動詞の時制ペア（現在↔過去）
_BE_TENSE_PAIRS = {
    frozenset(p) for p in [("is", "was"), ("are", "were"), ("am", "was")]
}

# 追加・削除されると時制変化を示唆する助動詞
_TENSE_MARKERS = {"will", "did", "had", "has", "have", "was", "were"}

# 三単現の主語（直前にあれば plural ではなく subject_verb と判定）
_SINGULAR_SUBJECTS = {"he", "she", "it", "this", "that"}


def _tokenize(text: str) -> list[str]:
    """英単語のみを小文字で抽出する"""
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def _is_tense_pair(a: str, b: str) -> bool:
    """2語が時制変化のペアかどうか（規則変化・不規則変化・be動詞）"""
    pair = frozenset((a, b))
    if pair in _IRREGULAR_PAST_PAIRS or pair in _BE_TENSE_PAIRS:
        return True
    # 規則変化: play → played / like → liked 等
    for x, y in ((a, b), (b, a)):
        if y == x + "ed" or y == x + "d":
            return True
        # study → studied のような y→ied 変化
        if x.endswith("y") and y == x[:-1] + "ied":
            return True
    return False


def _is_plural_pair(a: str, b: str) -> bool:
    """2語が単複変化のペアかどうか（末尾 s / es / ies）"""
    for x, y in ((a, b), (b, a)):
        if y == x + "s" or y == x + "es":
            return True
        if x.endswith("y") and y == x[:-1] + "ies":
            return True
    return False


def _diff_heuristic(original: str, corrected: str) -> str:
    """original/correctedの単語diffからエラー種別を推定する"""
    orig = _tokenize(original)
    corr = _tokenize(corrected)
    if not orig or not corr:
        return "other"

    # 語順: 単語の構成は同じで並びだけ違う
    if orig != corr and sorted(orig) == sorted(corr):
        return "word_order"

    # diffを取り、置換ペア・追加語・削除語を収集する
    sm = difflib.SequenceMatcher(a=orig, b=corr)
    replaced_pairs: list[tuple[str, str]] = []  # (元の語, 直した語)
    inserted: list[tuple[int, str]] = []        # (corrected内の位置, 語)
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            r, a = orig[i1:i2], corr[j1:j2]
            if len(r) == len(a):
                replaced_pairs.extend(zip(r, a))
            else:
                removed.extend(r)
                inserted.extend((j1 + k, w) for k, w in enumerate(a))
        elif tag == "delete":
            removed.extend(orig[i1:i2])
        elif tag == "insert":
            inserted.extend((j1 + k, w) for k, w in enumerate(corr[j1:j2]))

    added_words = [w for _, w in inserted]
    changed = removed + added_words + [w for pair in replaced_pairs for w in pair]

    # 主語動詞一致: is/are, has/have 等の置換ペア
    if any(frozenset(p) in _SV_PAIRS for p in replaced_pairs):
        return "subject_verb"

    # 三単現: s付き置換で直前の主語が he/she/it 等なら subject_verb
    for j, (a, b) in enumerate(replaced_pairs):
        if _is_plural_pair(a, b):
            idx = corr.index(b) if b in corr else -1
            if idx > 0 and corr[idx - 1] in _SINGULAR_SUBJECTS:
                return "subject_verb"

    # 時制: 規則/不規則変化ペア、または will/did 等の助動詞の増減
    if any(_is_tense_pair(a, b) for a, b in replaced_pairs):
        return "tense"
    if any(w in _TENSE_MARKERS for w in added_words + removed):
        return "tense"

    # 冠詞: a/an/the の追加・削除・置換
    if any(w in _ARTICLES for w in changed):
        return "article"

    # 前置詞: 前置詞同士の置換、または前置詞の追加・削除
    if any(a in _PREPOSITIONS and b in _PREPOSITIONS for a, b in replaced_pairs):
        return "preposition"
    if any(w in _PREPOSITIONS for w in added_words + removed):
        return "preposition"

    # 単複: 末尾 s/es/ies の変化
    if any(_is_plural_pair(a, b) for a, b in replaced_pairs):
        return "plural"

    # スペル: 置換ペアの文字列類似度が高ければタイプミスとみなす
    for a, b in replaced_pairs:
        if a != b and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8:
            return "spelling"

    # 語彙: 上記に該当しない単語の置き換え
    if replaced_pairs:
        return "vocabulary"

    return "other"


def tag_correction(original: str, corrected: str, explanation: str) -> str:
    """
    1件の添削からエラー種別を判定して返す。
    explanationのキーワード → diffヒューリスティック の順に評価し、
    最初にマッチしたカテゴリを返す（該当なしは "other"）。
    """
    text = explanation or ""
    for etype, pattern in _COMPILED_RULES:
        if pattern.search(text):
            return etype
    return _diff_heuristic(original or "", corrected or "")
