"""
AI英会話アプリケーション - バックエンドサーバー
FastAPI + Ollama/OpenAI対応
"""

import asyncio
import json
import locale
import os
import re
import subprocess
import tempfile
from pathlib import Path
from threading import Lock
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config

# ============================================================
# アプリケーション初期化
# ============================================================
app = FastAPI(title="AI English Conversation Tutor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会話履歴（セッション管理、簡易版）
conversations: dict[str, list] = {}
faster_whisper_model = None
faster_whisper_lock = Lock()

# ============================================================
# IPA発音記号データ（基本的な英単語の発音辞書）
# ============================================================
# CMU Pronouncing Dictionaryベースの簡易IPA変換
# eng_to_ipaライブラリが利用可能な場合はそちらを使用
try:
    import eng_to_ipa
    HAS_IPA_LIB = True
except ImportError:
    HAS_IPA_LIB = False

# ARPAbetからIPA変換マップ
ARPABET_TO_IPA = {
    'AA': 'ɑː', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔː', 'AW': 'aʊ',
    'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
    'EH': 'ɛ', 'ER': 'ɜːr', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ',
    'HH': 'h', 'IH': 'ɪ', 'IY': 'iː', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'uː', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}

# CMU辞書の読み込み
cmu_dict: dict[str, str] = {}

def load_cmu_dict():
    """CMU発音辞書を読み込む"""
    global cmu_dict
    try:
        import nltk
        nltk.download('cmudict', quiet=True)
        from nltk.corpus import cmudict
        entries = cmudict.entries()
        for word, phones in entries:
            ipa_phones = []
            for phone in phones:
                # ストレス番号を除去
                clean = re.sub(r'\d', '', phone)
                if clean in ARPABET_TO_IPA:
                    ipa_phones.append(ARPABET_TO_IPA[clean])
            cmu_dict[word.lower()] = ''.join(ipa_phones)
        print(f"CMU dictionary loaded: {len(cmu_dict)} words")
    except Exception as e:
        print(f"CMU dictionary load failed (using simple mode): {e}")


def get_ipa(word: str) -> str:
    """単語のIPA発音記号を取得"""
    word_lower = word.lower().strip(".,!?;:'\"")
    if not word_lower:
        return ""

    # eng_to_ipaライブラリがあればそちらを優先
    if HAS_IPA_LIB:
        result = eng_to_ipa.convert(word_lower)
        if result and '*' not in result:
            return result

    # CMU辞書から検索
    if word_lower in cmu_dict:
        return cmu_dict[word_lower]

    return ""


def get_sentence_ipa(sentence: str) -> list[dict]:
    """文全体のIPA発音記号を取得"""
    words = re.findall(r"[a-zA-Z']+|[.,!?;:]", sentence)
    result = []
    for word in words:
        ipa = get_ipa(word)
        result.append({
            "word": word,
            "ipa": ipa if ipa else None
        })
    return result


# ============================================================
# 発音スコアリング
# ============================================================
def calculate_pronunciation_score(
    recognized_text: str,
    expected_text: Optional[str] = None
) -> dict:
    """
    発音スコアを算出する。
    Web Speech APIの認識結果に基づくシンプルなスコアリング。
    """
    if not recognized_text.strip():
        return {"overall_score": 0, "details": "音声が検出されませんでした"}

    words = re.findall(r"[a-zA-Z']+", recognized_text.lower())

    if not words:
        return {"overall_score": 0, "details": "英語が検出されませんでした"}

    # 基本スコア（音声認識が成功した時点で60点以上）
    base_score = 65

    # 単語ごとのIPA取得率でスコア加算
    ipa_found = sum(1 for w in words if get_ipa(w))
    ipa_ratio = ipa_found / len(words) if words else 0

    # 文の長さや複雑さに基づくボーナス
    length_bonus = min(10, len(words) * 0.5)
    complexity_bonus = min(10, len(set(words)) * 0.3)

    # Web Speech APIの信頼度に基づくスコア
    # (ブラウザ側でconfidenceが取れる場合はフロントエンドから渡す)
    recognition_bonus = 15  # 認識成功のベースボーナス

    score = min(100, base_score + length_bonus + complexity_bonus + recognition_bonus)

    return {
        "overall_score": round(score),
        "word_count": len(words),
        "unique_words": len(set(words)),
        "details": "音声認識に成功しました"
    }


# ============================================================
# LLMクライアント
# ============================================================
async def call_ollama(messages: list[dict]) -> str:
    """Ollama API - tries /api/chat first, falls back to /api/generate"""
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        # --- Try /api/chat (Ollama >= 0.1.14) ---
        try:
            chat_payload = {
                "model": config.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7},
            }
            resp = await client.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json=chat_payload,
            )
            if resp.status_code != 404:
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to Ollama. Make sure Ollama is running.",
            )
        except httpx.HTTPStatusError:
            pass  # fall through to /api/generate

        # --- Fallback: /api/generate (older Ollama) ---
        try:
            # Convert messages to a single prompt string
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            prompt_parts.append("Assistant:")
            prompt = "\n\n".join(prompt_parts)

            gen_payload = {
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7},
            }
            resp = await client.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json=gen_payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to Ollama. Make sure Ollama is running.",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")


async def call_openai(messages: list[dict]) -> str:
    """OpenAI APIを呼び出す（将来の拡張用）"""
    import httpx

    if not config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured.",
        )

    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def call_anthropic(messages: list[dict]) -> str:
    """Anthropic Messages APIを呼び出す"""
    import httpx

    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Anthropic API key is not configured.",
        )

    system_parts = [msg["content"] for msg in messages if msg["role"] == "system"]
    anthropic_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg["role"] in ("user", "assistant")
    ]

    headers = {
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": config.ANTHROPIC_API_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": config.ANTHROPIC_MODEL,
        "max_tokens": config.ANTHROPIC_MAX_TOKENS,
        "messages": anthropic_messages,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = "Anthropic API error."
            try:
                err_data = e.response.json()
                detail = err_data.get("error", {}).get("message", detail)
            except Exception:
                if e.response.text:
                    detail = e.response.text[:300]
            raise HTTPException(status_code=502, detail=detail)
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Anthropic API error: {str(e)}",
            )

    data = resp.json()
    text_parts = [
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part).strip()


def get_llm_display_name() -> str:
    """UI表示用のLLM名"""
    if config.LLM_PROVIDER == "ollama":
        return f"ollama:{config.OLLAMA_MODEL}"
    if config.LLM_PROVIDER == "openai":
        return f"openai:{config.OPENAI_MODEL}"
    if config.LLM_PROVIDER == "anthropic":
        return f"anthropic:{config.ANTHROPIC_MODEL}"
    return config.LLM_PROVIDER


def is_llm_ready() -> bool:
    """現在のLLM設定が利用可能かどうか"""
    if config.LLM_PROVIDER == "ollama":
        return True
    if config.LLM_PROVIDER == "openai":
        return bool(config.OPENAI_API_KEY)
    if config.LLM_PROVIDER == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    return False


def is_faster_whisper_available() -> bool:
    """faster-whisperが利用可能かどうか"""
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def is_stt_ready() -> bool:
    """サーバー側STTが利用可能かどうか"""
    if config.STT_PROVIDER == "faster_whisper":
        return is_faster_whisper_available()
    if config.STT_PROVIDER == "openai":
        return bool(config.OPENAI_API_KEY)
    if config.STT_PROVIDER == "disabled":
        return True
    return False


def get_stt_display_name() -> str:
    """UI表示用のSTT名"""
    if config.STT_PROVIDER == "faster_whisper":
        return f"faster_whisper:{config.FASTER_WHISPER_MODEL}"
    if config.STT_PROVIDER == "openai":
        return f"openai:{config.OPENAI_WHISPER_MODEL}"
    return config.STT_PROVIDER


def resolve_faster_whisper_runtime() -> tuple[str, str]:
    """実行環境に応じたfaster-whisperのdevice/compute_typeを決める"""
    import ctranslate2

    device = config.FASTER_WHISPER_DEVICE
    compute_type = config.FASTER_WHISPER_COMPUTE_TYPE

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

    if compute_type == "auto":
        supported = ctranslate2.get_supported_compute_types(device)
        if device == "cuda":
            for candidate in ("int8_float16", "float16", "int8", "float32"):
                if candidate in supported:
                    compute_type = candidate
                    break
        else:
            for candidate in ("int8", "int8_float32", "float32"):
                if candidate in supported:
                    compute_type = candidate
                    break

    return device, compute_type


def get_faster_whisper_model():
    """遅延ロードでfaster-whisperモデルを取得"""
    global faster_whisper_model

    if faster_whisper_model is not None:
        return faster_whisper_model

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"faster-whisper is not available: {str(e)}",
        )

    with faster_whisper_lock:
        if faster_whisper_model is not None:
            return faster_whisper_model

        device, compute_type = resolve_faster_whisper_runtime()
        faster_whisper_model = WhisperModel(
            config.FASTER_WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
            cpu_threads=config.FASTER_WHISPER_CPU_THREADS,
            num_workers=config.FASTER_WHISPER_NUM_WORKERS,
            download_root=config.FASTER_WHISPER_DOWNLOAD_ROOT,
            local_files_only=config.FASTER_WHISPER_LOCAL_FILES_ONLY,
        )
        print(
            "Loaded faster-whisper model "
            f"({config.FASTER_WHISPER_MODEL}, device={device}, compute_type={compute_type})"
        )

    return faster_whisper_model


def guess_audio_extension(content_type: Optional[str]) -> str:
    """録音MIME typeから拡張子を推定"""
    mapping = {
        "audio/webm": ".webm",
        "video/webm": ".webm",
        "audio/mp4": ".m4a",
        "video/mp4": ".mp4",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/m4a": ".m4a",
    }
    return mapping.get((content_type or "").lower(), ".webm")


async def transcribe_audio_openai(
    audio_bytes: bytes,
    filename: str,
    content_type: Optional[str],
) -> dict:
    """OpenAI Audio APIで音声を文字起こしする"""
    import httpx

    if not config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server-side transcription is not configured. Set OPENAI_API_KEY in the environment or .env.",
        )

    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
    files = {
        "file": (
            filename,
            audio_bytes,
            content_type or "application/octet-stream",
        )
    }
    data = {
        "model": config.OPENAI_WHISPER_MODEL,
        "language": config.OPENAI_TRANSCRIPTION_LANGUAGE,
        "response_format": "json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = "OpenAI transcription failed."
            try:
                err_data = e.response.json()
                detail = err_data.get("error", {}).get("message", detail)
            except Exception:
                if e.response.text:
                    detail = e.response.text[:300]
            raise HTTPException(status_code=502, detail=detail)
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI transcription failed: {str(e)}",
            )

    payload = resp.json()
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Transcription returned an empty result.")

    return {"text": text, "provider": "openai"}


def _transcribe_audio_faster_whisper_file(audio_path: str) -> dict:
    """同期処理: faster-whisperでファイルを文字起こし"""
    model = get_faster_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        task="transcribe",
        language=config.FASTER_WHISPER_LANGUAGE or None,
        beam_size=config.FASTER_WHISPER_BEAM_SIZE,
        vad_filter=config.FASTER_WHISPER_VAD_FILTER,
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()

    if not text:
        raise HTTPException(status_code=422, detail="No speech detected. Try again.")

    return {
        "text": text,
        "provider": "faster_whisper",
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
    }


async def transcribe_audio_faster_whisper(
    audio_bytes: bytes,
    filename: str,
    content_type: Optional[str],
) -> dict:
    """ローカル faster-whisper で音声を文字起こしする"""
    suffix = Path(filename).suffix or guess_audio_extension(content_type)
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        return await asyncio.to_thread(_transcribe_audio_faster_whisper_file, temp_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Local Whisper transcription failed: {str(e)}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


async def get_llm_response(messages: list[dict]) -> str:
    """設定に基づいてLLMを呼び出す"""
    if config.LLM_PROVIDER == "ollama":
        return await call_ollama(messages)
    elif config.LLM_PROVIDER == "openai":
        return await call_openai(messages)
    elif config.LLM_PROVIDER == "anthropic":
        return await call_anthropic(messages)
    else:
        raise HTTPException(status_code=500, detail=f"未対応のLLMプロバイダー: {config.LLM_PROVIDER}")


# ============================================================
# LLM JSON Parser (robust)
# ============================================================
def parse_llm_json(raw: str) -> dict:
    """
    LLMの応答からJSON部分を確実に抽出する。
    マークダウンコードブロック、余計なテキスト、ネストJSON等に対応。
    """
    fallback = {
        "reply": "",
        "corrections": [],
        "natural_expression": None,
        "encouragement": "",
    }

    if not raw or not raw.strip():
        fallback["reply"] = "Sorry, I didn't get a response. Please try again."
        return fallback

    # 1. Strip markdown code block: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r'```(?:json)?\s*', '', raw)
    cleaned = re.sub(r'```', '', cleaned).strip()

    # 2. Try to find and parse a JSON object with "reply" key
    # Use a non-greedy approach: find all { ... } candidates
    candidates = []

    # Find the outermost { ... } that contains "reply"
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(cleaned[start:i+1])
                start = -1

    # Try each candidate, prefer the one with "reply" key
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "reply" in obj:
                # Validate reply is a plain string, not JSON
                reply = obj.get("reply", "")
                if isinstance(reply, str) and reply.strip().startswith('{'):
                    try:
                        inner = json.loads(reply)
                        if isinstance(inner, dict) and "reply" in inner:
                            return inner  # unwrap nested JSON
                    except json.JSONDecodeError:
                        pass
                return obj
        except json.JSONDecodeError:
            continue

    # 3. Fallback: treat the entire response as plain text reply
    # Strip any JSON artifacts that leaked through
    plain = raw.strip()
    # Remove wrapping JSON if it looks like the raw JSON was the reply
    if plain.startswith('{') and plain.endswith('}'):
        try:
            obj = json.loads(plain)
            if isinstance(obj, dict) and "reply" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Final fallback: use raw text as the reply
    fallback["reply"] = plain
    return fallback


# ============================================================
# APIモデル
# ============================================================
class ChatRequest(BaseModel):
    text: str
    session_id: str = "default"
    confidence: float = 0.0  # Web Speech APIの認識信頼度

class TTSRequest(BaseModel):
    text: str


# ============================================================
# APIエンドポイント
# ============================================================
@app.on_event("startup")
async def startup():
    """サーバー起動時の初期化"""
    load_cmu_dict()
    print(f"\n{'='*50}")
    print(f"  AI English Conversation Tutor - Starting...")
    print(f"  LLM Provider: {config.LLM_PROVIDER}")
    if config.LLM_PROVIDER == "ollama":
        print(f"  Ollama Model: {config.OLLAMA_MODEL}")
    elif config.LLM_PROVIDER == "openai":
        print(f"  OpenAI Model: {config.OPENAI_MODEL}")
    elif config.LLM_PROVIDER == "anthropic":
        print(f"  Anthropic Model: {config.ANTHROPIC_MODEL}")
    print(f"  STT Provider: {config.STT_PROVIDER}")
    if config.STT_PROVIDER == "faster_whisper":
        print(f"  Whisper Model: {config.FASTER_WHISPER_MODEL}")
    print(f"{'='*50}\n")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    メインの会話エンドポイント
    - ユーザーの英語テキストを受け取る
    - LLMで添削・応答を生成
    - IPA発音記号を付与
    - 発音スコアを算出
    """
    user_text = req.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="テキストが空です")

    # セッション管理
    if req.session_id not in conversations:
        conversations[req.session_id] = []

    history = conversations[req.session_id]

    # メッセージ構築
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    # 直近の会話履歴を追加
    for msg in history[-config.MAX_CONVERSATION_HISTORY:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_text})

    # LLM呼び出し
    raw_response = await get_llm_response(messages)

    # JSON応答のパース
    ai_response = parse_llm_json(raw_response)

    # 会話履歴に追加
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": ai_response.get("reply", raw_response)})

    # 発音記号を取得
    user_ipa = get_sentence_ipa(user_text)
    reply_ipa = get_sentence_ipa(ai_response.get("reply", ""))

    # 発音スコアを算出
    pronunciation_score = calculate_pronunciation_score(user_text)

    # 認識信頼度があれば反映
    if req.confidence > 0:
        confidence_factor = req.confidence * 100
        pronunciation_score["overall_score"] = round(
            pronunciation_score["overall_score"] * 0.5 + confidence_factor * 0.5
        )
        pronunciation_score["recognition_confidence"] = round(req.confidence * 100)

    return JSONResponse({
        "reply": ai_response.get("reply", raw_response),
        "corrections": ai_response.get("corrections", []),
        "natural_expression": ai_response.get("natural_expression"),
        "encouragement": ai_response.get("encouragement", ""),
        "user_ipa": user_ipa,
        "reply_ipa": reply_ipa,
        "pronunciation_score": pronunciation_score,
    })


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """録音音声をサーバー側で文字起こしする"""
    audio_bytes = await audio.read()
    try:
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file is empty.")

        if len(audio_bytes) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio file is too large. Keep it under 25 MB.")

        content_type = audio.content_type or "application/octet-stream"
        filename = audio.filename or f"recording{guess_audio_extension(content_type)}"

        if config.STT_PROVIDER == "faster_whisper":
            result = await transcribe_audio_faster_whisper(audio_bytes, filename, content_type)
        elif config.STT_PROVIDER == "openai":
            result = await transcribe_audio_openai(audio_bytes, filename, content_type)
        elif config.STT_PROVIDER == "disabled":
            raise HTTPException(status_code=503, detail="Server-side transcription is disabled.")
        else:
            raise HTTPException(status_code=500, detail=f"未対応のSTTプロバイダー: {config.STT_PROVIDER}")

        return JSONResponse(result)
    finally:
        await audio.close()


@app.post("/api/pronunciation")
async def get_pronunciation(req: TTSRequest):
    """テキストのIPA発音記号のみを取得"""
    ipa_data = get_sentence_ipa(req.text)
    return JSONResponse({"ipa": ipa_data})


@app.get("/api/health")
async def health():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "provider": config.LLM_PROVIDER,
        "llm_model": get_llm_display_name(),
        "llm_ready": is_llm_ready(),
        "stt_provider": config.STT_PROVIDER,
        "stt_model": get_stt_display_name(),
        "stt_ready": is_stt_ready(),
    }


@app.get("/api/models")
async def list_models():
    """利用可能なOllamaモデルを一覧"""
    if config.LLM_PROVIDER == "openai":
        return {"models": [config.OPENAI_MODEL], "current": config.OPENAI_MODEL}
    if config.LLM_PROVIDER == "anthropic":
        return {"models": [config.ANTHROPIC_MODEL], "current": config.ANTHROPIC_MODEL}
    if config.LLM_PROVIDER != "ollama":
        return {"models": [], "current": None}

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"models": models, "current": config.OLLAMA_MODEL}
    except Exception:
        return {"models": [], "current": config.OLLAMA_MODEL, "error": "Ollamaに接続できません"}


# ============================================================
# 静的ファイル配信
# ============================================================
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
# Network helpers / SSL証明書の自動生成
# ============================================================
def parse_windows_ipconfig() -> list[tuple[str, str]]:
    """Windows の ipconfig 出力から (adapter, ipv4) を抽出する"""
    try:
        result = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="ignore",
            check=True,
        )
    except Exception:
        return []

    entries: list[tuple[str, str]] = []
    current_adapter = None
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if not line.startswith(" ") and "adapter" in stripped.lower() and stripped.endswith(":"):
            current_adapter = stripped[:-1]
            continue

        if "IPv4 Address" in stripped or "IPv4 アドレス" in stripped:
            match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", stripped)
            if match and current_adapter:
                entries.append((current_adapter, match.group(1)))

    return entries


def get_tailscale_status() -> dict:
    """tailscale status --json を取得する"""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )
        return json.loads(result.stdout)
    except Exception:
        return {}


def collect_tailscale_dns_names() -> list[str]:
    """MagicDNS で使える Tailscale 名を集める"""
    status = get_tailscale_status()
    self_data = status.get("Self", {})

    names = set()
    dns_name = (self_data.get("DNSName") or "").strip().rstrip(".")
    host_name = (self_data.get("HostName") or "").strip()

    if dns_name:
        names.add(dns_name)
        short_name = dns_name.split(".", 1)[0]
        if short_name:
            names.add(short_name)

    if host_name:
        names.add(host_name)
        names.add(host_name.lower())

    return sorted(name for name in names if name)


def collect_network_ipv4_addresses() -> list[str]:
    """証明書SANや接続案内に使う IPv4 を集める"""
    import ipaddress
    import socket

    ips: set[str] = {"127.0.0.1"}
    hostnames = {socket.gethostname(), socket.getfqdn(), "localhost"}

    for name in hostnames:
        if not name:
            continue
        try:
            for result in socket.getaddrinfo(name, None, socket.AF_INET):
                ip = result[4][0]
                if ip and ip != "127.0.0.1":
                    ips.add(ip)
        except OSError:
            continue

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass

    for _adapter, ip in parse_windows_ipconfig():
        ips.add(ip)

    valid_ips = []
    for ip in ips:
        try:
            parsed = ipaddress.ip_address(ip)
            if parsed.version == 4:
                valid_ips.append(str(parsed))
        except ValueError:
            continue

    return sorted(valid_ips, key=lambda ip: tuple(int(part) for part in ip.split(".")))


def collect_certificate_dns_names() -> list[str]:
    """証明書に入れるDNS名を集める"""
    import socket

    names = {"localhost", socket.gethostname(), socket.getfqdn()}
    names.update(collect_tailscale_dns_names())
    cleaned = [name.strip() for name in names if name and name.strip()]
    return sorted(set(cleaned))


def collect_access_urls() -> list[tuple[str, str]]:
    """起動時に表示するアクセスURLを作る"""
    urls: list[tuple[str, str]] = [("PC", f"https://localhost:{config.PORT}")]

    tailscale_names = collect_tailscale_dns_names()
    fqdn = next((name for name in tailscale_names if ".ts.net" in name), None)
    if fqdn:
        urls.append(("VPN DNS", f"https://{fqdn}:{config.PORT}"))

    for adapter, ip in parse_windows_ipconfig():
        adapter_lower = adapter.lower()
        if "tailscale" in adapter_lower:
            label = "VPN"
        elif "ethernet" in adapter_lower or "イーサネット" in adapter or "wi-fi" in adapter_lower or "wifi" in adapter_lower:
            label = "LAN"
        else:
            label = "Network"
        urls.append((label, f"https://{ip}:{config.PORT}"))

    seen = set()
    deduped = []
    for label, url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append((label, url))
    return deduped


def existing_cert_covers_targets(certfile: str, dns_names: list[str], ip_addresses: list[str]) -> bool:
    """既存証明書のSANに必要な DNS / IP が揃っているか確認する"""
    import ssl

    try:
        cert = ssl._ssl._test_decode_cert(certfile)
    except Exception:
        return False

    san_entries = cert.get("subjectAltName", ())
    existing_dns = {value for kind, value in san_entries if kind == "DNS"}
    existing_ips = {value for kind, value in san_entries if kind == "IP Address"}
    return set(dns_names).issubset(existing_dns) and set(ip_addresses).issubset(existing_ips)


def generate_self_signed_cert():
    """Generate self-signed SSL cert using Python (no openssl CLI needed)"""
    dns_names = collect_certificate_dns_names()
    ip_addresses = ["0.0.0.0"] + [
        ip for ip in collect_network_ipv4_addresses()
        if ip != "0.0.0.0"
    ]

    cert_exists = os.path.exists(config.SSL_CERTFILE) and os.path.exists(config.SSL_KEYFILE)
    if cert_exists and existing_cert_covers_targets(config.SSL_CERTFILE, dns_names, ip_addresses):
        print("Using existing SSL certificate")
        return
    if cert_exists:
        print("Regenerating SSL certificate to include current network addresses...")

    print("Generating self-signed SSL certificate...")
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        import ipaddress

        # Generate RSA key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Build certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "AI-English-Tutor"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Local"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "JP"),
        ])

        san_list = [x509.DNSName(name) for name in dns_names]
        san_list.extend(
            x509.IPAddress(ipaddress.IPv4Address(ip))
            for ip in ip_addresses
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
            .sign(key, hashes.SHA256())
        )

        # Write key file
        with open(config.SSL_KEYFILE, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))

        # Write cert file
        with open(config.SSL_CERTFILE, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print(
            "SSL certificate generated "
            f"(SAN includes: {', '.join(ip_addresses)})"
        )

    except ImportError:
        print("ERROR: 'cryptography' package not found.")
        print("Run:  pip install cryptography")
        print("Microphone will NOT work on mobile without HTTPS.")
    except Exception as e:
        print(f"SSL certificate generation failed: {e}")


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    generate_self_signed_cert()

    print(f"\n{'='*50}")
    print(f"  AI English Conversation Tutor")
    print(f"  ")
    for label, url in collect_access_urls():
        print(f"  {label:<10} {url}")
    print(f"  ")
    print(f"  * On first mobile access, you will see a")
    print(f"    security warning. Tap 'Advanced' then")
    print(f"    'Proceed' to continue (self-signed cert).")
    print(f"{'='*50}\n")

    use_ssl = os.path.exists(config.SSL_CERTFILE) and os.path.exists(config.SSL_KEYFILE)

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        ssl_certfile=config.SSL_CERTFILE if use_ssl else None,
        ssl_keyfile=config.SSL_KEYFILE if use_ssl else None,
    )
