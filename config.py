"""
AI英会話アプリケーション設定ファイル
ローカル/クラウドLLMとローカル/クラウドSTTを切り替え可能
"""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_local_env() -> None:
    """Simple .env loader to keep API keys out of source files."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


load_local_env()

# =============================================================
# LLMプロバイダー設定 ("ollama" / "openai" / "anthropic")
# =============================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- Ollama設定 ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:latest")  # ollama list で表示されるモデル名に合わせる

# --- Anthropic設定 ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 1024

# --- OpenAI設定 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # sk-... の形式
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
OPENAI_WHISPER_MODEL = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
OPENAI_TRANSCRIPTION_LANGUAGE = os.getenv("OPENAI_TRANSCRIPTION_LANGUAGE", "en")

# --- 音声認識設定 ---
STT_PROVIDER = os.getenv("STT_PROVIDER", "faster_whisper")  # "faster_whisper" / "openai" / "disabled"

# --- faster-whisper設定（ローカルSTT） ---
FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "base.en")
FASTER_WHISPER_LANGUAGE = os.getenv("FASTER_WHISPER_LANGUAGE", "en")
FASTER_WHISPER_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "auto")
FASTER_WHISPER_COMPUTE_TYPE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "auto")
FASTER_WHISPER_CPU_THREADS = int(os.getenv("FASTER_WHISPER_CPU_THREADS", "0"))
FASTER_WHISPER_NUM_WORKERS = int(os.getenv("FASTER_WHISPER_NUM_WORKERS", "1"))
FASTER_WHISPER_BEAM_SIZE = int(os.getenv("FASTER_WHISPER_BEAM_SIZE", "1"))
FASTER_WHISPER_VAD_FILTER = os.getenv("FASTER_WHISPER_VAD_FILTER", "true").lower() in {"1", "true", "yes", "on"}
FASTER_WHISPER_DOWNLOAD_ROOT = os.getenv("FASTER_WHISPER_DOWNLOAD_ROOT", "") or None
FASTER_WHISPER_LOCAL_FILES_ONLY = os.getenv("FASTER_WHISPER_LOCAL_FILES_ONLY", "false").lower() in {"1", "true", "yes", "on"}

# =============================================================
# サーバー設定
# =============================================================
HOST = "0.0.0.0"
PORT = 8443
# HTTPS用自己署名証明書（初回起動時に自動生成）
SSL_CERTFILE = "cert.pem"
SSL_KEYFILE = "key.pem"

# =============================================================
# アプリケーション設定
# =============================================================
# 会話履歴の最大保持数
MAX_CONVERSATION_HISTORY = 20

# システムプロンプト（英会話講師の役割）
SYSTEM_PROMPT = """You are a strict English grammar tutor. Keep replies concise (1-2 sentences).

STEP 1 - FIND ALL ERRORS (do this carefully before writing your reply):
Look for these common errors in the student's message:
- Wrong word order (e.g. "what do you think is it important" should be "do you think it is important")
- Duplicated words (e.g. "is it it" has an extra "it")
- Missing articles (a/an/the)
- Wrong prepositions
- Subject-verb agreement
- Tense errors
- Any other grammar mistakes
You MUST catch and correct ALL errors. Never skip an error.

STEP 2 - REPLY:
Reply naturally in English. Ask a follow-up question to continue the conversation.
ALL text must be in English only. No Japanese or non-Latin characters.

STEP 3 - OUTPUT FORMAT:
Respond ONLY with this JSON. No other text:
{"reply":"your English response","corrections":[{"original":"the exact wrong phrase","corrected":"the corrected phrase","explanation":"why this is wrong"}],"natural_expression":"a more natural way to say the whole sentence, or null if already natural"}

Examples:
Student: "what do you think is it it important to study"
{"reply":"Yes, I think studying is very important! What subject are you most interested in?","corrections":[{"original":"what do you think is it it important","corrected":"do you think it is important","explanation":"Wrong word order and duplicated 'it'. Use: Do you think + subject + verb"},{"original":"is it it","corrected":"it is","explanation":"Remove the extra 'it' and fix word order"}],"natural_expression":"Do you think it's important to study?"}

Student: "I go to school yesterday"
{"reply":"What did you do at school?","corrections":[{"original":"I go to school yesterday","corrected":"I went to school yesterday","explanation":"Use past tense 'went' with 'yesterday'"}],"natural_expression":null}

Student: "I had a great time at the party last night"
{"reply":"That sounds fun! What was the best part?","corrections":[],"natural_expression":null}"""
