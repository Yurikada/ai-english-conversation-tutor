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
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import db
from review_logic import judge_review_answer

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
# 単語レベル判定の閾値（faster-whisperの単語別認識確率に対する基準）
WORD_LEVEL_GOOD_THRESHOLD = 0.85   # これ以上は good（明瞭に発音できている）
WORD_LEVEL_FAIR_THRESHOLD = 0.60   # これ以上は fair、未満は poor（要練習）

# 低確率単語へのペナルティ（1語あたりの減点）
POOR_WORD_PENALTY = 5
FAIR_WORD_PENALTY = 2


def score_word_level(probability: float) -> str:
    """単語の認識確率を good / fair / poor のレベルに変換する"""
    if probability >= WORD_LEVEL_GOOD_THRESHOLD:
        return "good"
    if probability >= WORD_LEVEL_FAIR_THRESHOLD:
        return "fair"
    return "poor"


def calculate_pronunciation_score_acoustic(words: list) -> Optional[dict]:
    """
    faster-whisperの単語別認識確率に基づく音響ベースの発音スコア。
    - 平均確率を主成分とする（平均prob 0.95 → 95点相当）
    - 低確率単語（poor/fair）の数に応じてペナルティを加える
    有効な単語情報が1つも無い場合は None を返し、呼び出し側でフォールバックする。
    """
    word_scores = []
    probs = []
    for entry in words or []:
        # フロント経由のJSONなのでdict以外や欠損値は黙って読み飛ばす
        if not isinstance(entry, dict):
            continue
        word = str(entry.get("word", "")).strip().strip(".,!?;:\"")
        try:
            prob = float(entry.get("probability"))
        except (TypeError, ValueError):
            continue
        if not word or not (0.0 <= prob <= 1.0):
            continue
        word_scores.append({
            "word": word,
            "score": round(prob * 100),
            "level": score_word_level(prob),
        })
        probs.append(prob)

    if not probs:
        return None

    # 平均確率を100点満点に換算し、低確率単語数でペナルティ
    avg_prob = sum(probs) / len(probs)
    poor_count = sum(1 for ws in word_scores if ws["level"] == "poor")
    fair_count = sum(1 for ws in word_scores if ws["level"] == "fair")
    penalty = poor_count * POOR_WORD_PENALTY + fair_count * FAIR_WORD_PENALTY
    overall = max(0, min(100, round(avg_prob * 100 - penalty)))

    details = f"音響解析: 平均認識確率 {round(avg_prob * 100)}%"
    if poor_count:
        details += f" / 要練習 {poor_count}語"

    return {
        "overall_score": overall,
        "word_count": len(word_scores),
        "unique_words": len({ws["word"].lower() for ws in word_scores}),
        "details": details,
        "method": "acoustic",
        "average_probability": round(avg_prob, 3),
        "word_scores": word_scores,
    }


def calculate_pronunciation_score(
    recognized_text: str,
    expected_text: Optional[str] = None,
    words: Optional[list] = None,
) -> dict:
    """
    発音スコアを算出する。
    - words（faster-whisperの単語別確率）があれば音響ベース（method: "acoustic"）
    - 無い場合（テキスト入力・OpenAI STT）は従来型の簡易スコア（method: "text_only"）
    """
    if not recognized_text.strip():
        return {"overall_score": 0, "details": "音声が検出されませんでした", "method": "text_only"}

    # 音響ベースのスコアリング（単語別確率がある場合）
    acoustic = calculate_pronunciation_score_acoustic(words)
    if acoustic is not None:
        return acoustic

    text_words = re.findall(r"[a-zA-Z']+", recognized_text.lower())

    if not text_words:
        return {"overall_score": 0, "details": "英語が検出されませんでした", "method": "text_only"}

    # --- 従来型フォールバック: テキストのみの簡易スコア ---
    # 基本スコア（音声認識が成功した時点で60点以上）
    base_score = 65

    # 文の長さや複雑さに基づくボーナス
    length_bonus = min(10, len(text_words) * 0.5)
    complexity_bonus = min(10, len(set(text_words)) * 0.3)

    # Web Speech APIの信頼度に基づくスコア
    # (ブラウザ側でconfidenceが取れる場合はフロントエンドから渡す)
    recognition_bonus = 15  # 認識成功のベースボーナス

    score = min(100, base_score + length_bonus + complexity_bonus + recognition_bonus)

    return {
        "overall_score": round(score),
        "word_count": len(text_words),
        "unique_words": len(set(text_words)),
        "details": "音声認識に成功しました（テキストのみの簡易評価）",
        "method": "text_only",
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


# ============================================================
# TTS（音声合成）
# ============================================================
# /api/tts のテキスト長上限（長文はブラウザ側TTSに任せる想定）
TTS_MAX_TEXT_LENGTH = 1000


def is_edge_tts_available() -> bool:
    """edge-ttsライブラリが利用可能かどうか"""
    try:
        import edge_tts  # noqa: F401
        return True
    except Exception:
        return False


def is_tts_ready() -> bool:
    """サーバー側TTSが利用可能かどうか"""
    if config.TTS_PROVIDER == "edge":
        return is_edge_tts_available()
    if config.TTS_PROVIDER == "openai":
        return bool(config.OPENAI_API_KEY)
    if config.TTS_PROVIDER == "browser":
        return True  # クライアント側TTSに委譲するので常にOK扱い
    return False


async def synthesize_speech_edge(text: str, voice: str) -> bytes:
    """edge-ttsでテキストからMP3バイト列を生成する"""
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="edge-tts is not installed. Run: pip install edge-tts",
        )

    communicate = edge_tts.Communicate(
        text,
        voice or config.EDGE_TTS_VOICE,
        rate=config.EDGE_TTS_RATE,
    )

    # ストリーミングでaudioチャンクを集めてMP3を組み立てる
    audio_chunks = []
    try:
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                audio_chunks.append(chunk["data"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"edge-tts synthesis failed: {str(e)}")

    audio = b"".join(audio_chunks)
    if not audio:
        raise HTTPException(status_code=502, detail="edge-tts returned no audio.")
    return audio


async def synthesize_speech_openai(text: str, voice: str) -> bytes:
    """OpenAI Audio Speech APIでテキストからMP3バイト列を生成する"""
    import httpx

    if not config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server-side TTS is not configured. Set OPENAI_API_KEY in the environment or .env.",
        )

    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.OPENAI_TTS_MODEL,
        "voice": voice or config.OPENAI_TTS_VOICE,
        "input": text,
        "response_format": "mp3",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = "OpenAI TTS failed."
            try:
                err_data = e.response.json()
                detail = err_data.get("error", {}).get("message", detail)
            except Exception:
                if e.response.text:
                    detail = e.response.text[:300]
            raise HTTPException(status_code=502, detail=detail)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"OpenAI TTS failed: {str(e)}")

    if not resp.content:
        raise HTTPException(status_code=502, detail="OpenAI TTS returned no audio.")
    return resp.content


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
    """同期処理: faster-whisperでファイルを文字起こし（単語別確率つき）"""
    model = get_faster_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        task="transcribe",
        language=config.FASTER_WHISPER_LANGUAGE or None,
        beam_size=config.FASTER_WHISPER_BEAM_SIZE,
        vad_filter=config.FASTER_WHISPER_VAD_FILTER,
        condition_on_previous_text=False,
        word_timestamps=True,  # 発音スコアリング用に単語別の確率・時刻を取得
    )

    texts = []
    words = []
    for segment in segments:
        if segment.text:
            texts.append(segment.text.strip())
        # 単語別の認識確率を収集（発音スコアの主成分になる）
        for w in (segment.words or []):
            word = (w.word or "").strip()
            if not word:
                continue
            words.append({
                "word": word,
                "probability": round(float(w.probability), 4),
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
            })

    text = " ".join(texts).strip()

    if not text:
        raise HTTPException(status_code=422, detail="No speech detected. Try again.")

    return {
        "text": text,
        "provider": "faster_whisper",
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "words": words,
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
# セッションサマリ（今日の学びレポート）
# ============================================================
# LLM講評用のシステムプロンプト（このエンドポイント専用・1回だけ呼ぶ）
SUMMARY_FEEDBACK_PROMPT = """You are an encouraging English tutor writing a short end-of-session report for a Japanese learner.
You will receive the session's conversation and the list of corrections made.

Respond ONLY with this JSON. No other text:
{"highlights": ["two specific things the student did well"], "focus_areas": ["two specific things to focus on next"], "phrase_to_remember": "one useful English phrase from this session worth memorizing", "message": "one short encouraging closing message in English"}

Rules:
- highlights and focus_areas must each have exactly 2 short items.
- Base everything on the actual session content. Keep each item under 20 words.
- ALL text must be in English only."""


def _summary_str_list(value, limit: int) -> list[str]:
    """LLM講評の値をstrのリストへ正規化する（不正型は空リスト）"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items = []
    for v in value:
        if isinstance(v, (str, int, float)):
            text = str(v).strip()
            if text:
                items.append(text)
    return items[:limit]


def parse_summary_llm_json(raw: str) -> dict:
    """
    /api/summary 用のLLM講評JSONを頑健に抽出する。
    parse_llm_json は "reply" キー前提の実装なのでここでは使わない。
    - マークダウンコードブロックを除去
    - 括弧の対応を取りながら {...} 候補を順に json.loads
    - 失敗時は highlights 等が空のフォールバックを返す
    """
    fallback = {
        "highlights": [],
        "focus_areas": [],
        "phrase_to_remember": None,
        "message": "",
    }

    if not raw or not raw.strip():
        return dict(fallback)

    # コードブロック記号（```json / ```）を除去
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # 括弧の深さを追いながら最初から順に {...} 候補を集める
    candidates = []
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(cleaned[start:i + 1])
                    start = -1

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        # 期待キーを型正規化して返す（余計なキーは捨てる）
        phrase = obj.get("phrase_to_remember")
        if not isinstance(phrase, str) or not phrase.strip():
            phrase = None
        message = obj.get("message")
        if not isinstance(message, str):
            message = ""
        return {
            "highlights": _summary_str_list(obj.get("highlights"), 2),
            "focus_areas": _summary_str_list(obj.get("focus_areas"), 2),
            "phrase_to_remember": phrase.strip() if phrase else None,
            "message": message.strip(),
        }

    # どの候補もパースできなかった場合のフォールバック
    return dict(fallback)


def build_session_summary_stats(data: dict) -> dict:
    """セッションサマリの統計部分をルールベースで組み立てる（LLM不使用）"""
    turns = data.get("turns", [])
    corrections = data.get("corrections", [])

    # 発音スコア（NULLは除外して集計）
    scores = [
        t["pronunciation_score"]
        for t in turns
        if isinstance(t.get("pronunciation_score"), (int, float))
    ]

    # エラー種別の内訳（未設定は 'other' に正規化、件数降順）
    error_counts: dict[str, int] = {}
    for c in corrections:
        error_type = (c.get("error_type") or "").strip() or "other"
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
    error_types = [
        {"error_type": et, "count": count}
        for et, count in sorted(error_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    # 学んだ自然な表現（重複は大文字小文字を無視して除去・登場順）
    seen = set()
    natural_expressions = []
    for t in turns:
        expression = (t.get("natural_expression") or "").strip()
        if expression and expression.lower() not in seen:
            seen.add(expression.lower())
            natural_expressions.append(expression)

    return {
        "turn_count": len(turns),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        "max_score": max(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "correction_count": len(corrections),
        "error_types": error_types,
        "natural_expressions": natural_expressions,
    }


def build_summary_feedback_messages(data: dict) -> list[dict]:
    """LLM講評用のメッセージ（会話内容 + 間違いリスト）を組み立てる"""
    lines = ["# Conversation"]
    # プロンプト肥大を防ぐため、ターン数・文字数を軽く制限する
    for t in data.get("turns", [])[:30]:
        lines.append(f"Student: {(t.get('user_text') or '')[:200]}")
        lines.append(f"Tutor: {(t.get('reply') or '')[:200]}")

    lines.append("")
    lines.append("# Corrections made during the session")
    corrections = data.get("corrections", [])
    if corrections:
        for c in corrections[:30]:
            error_type = (c.get("error_type") or "").strip() or "other"
            lines.append(
                f"- \"{(c.get('original') or '')[:120]}\" -> "
                f"\"{(c.get('corrected') or '')[:120]}\" ({error_type})"
            )
    else:
        lines.append("- (no corrections; the student spoke accurately)")

    return [
        {"role": "system", "content": SUMMARY_FEEDBACK_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


# ============================================================
# APIモデル
# ============================================================
class ChatRequest(BaseModel):
    text: str
    session_id: str = "default"
    confidence: float = 0.0  # Web Speech APIの認識信頼度
    words: list = Field(default_factory=list)  # /api/transcribe由来の単語別確率（任意）

class TTSRequest(BaseModel):
    text: str
    voice: str = ""  # 任意: プロバイダー既定のボイスを上書きする


class ReviewAnswerRequest(BaseModel):
    correction_id: int
    answer_text: str


# ============================================================
# APIエンドポイント
# ============================================================
@app.on_event("startup")
async def startup():
    """サーバー起動時の初期化"""
    db.init_db()
    print(f"SQLite database ready: {db.DB_PATH}")
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

    # 直近の会話履歴をDBから取得
    history = db.get_history(req.session_id, config.MAX_CONVERSATION_HISTORY)

    # メッセージ構築
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    # LLM呼び出し
    raw_response = await get_llm_response(messages)

    # JSON応答のパース
    ai_response = parse_llm_json(raw_response)
    if not ai_response.get("reply"):
        ai_response["reply"] = raw_response

    # 発音記号を取得
    user_ipa = get_sentence_ipa(user_text)
    reply_ipa = get_sentence_ipa(ai_response.get("reply", ""))

    # 発音スコアを算出（単語別確率があれば音響ベース、無ければテキストのみ）
    pronunciation_score = calculate_pronunciation_score(user_text, words=req.words)

    # 認識信頼度があれば反映（音響ベースのスコアには適用しない）
    if req.confidence > 0 and pronunciation_score.get("method") != "acoustic":
        confidence_factor = req.confidence * 100
        pronunciation_score["overall_score"] = round(
            pronunciation_score["overall_score"] * 0.5 + confidence_factor * 0.5
        )
        pronunciation_score["recognition_confidence"] = round(req.confidence * 100)

    # 会話ターンをDBに保存（履歴・添削・スコアの永続化）
    db.save_turn(
        req.session_id,
        user_text,
        ai_response,
        pronunciation_score.get("overall_score", 0),
    )

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


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """テキストをサーバー側で音声合成してMP3を返す
    - provider=edge: edge-tts（無料・要ネットワーク）
    - provider=openai: OpenAI Audio Speech API
    - provider=browser: 503を返してクライアント側TTSにフォールバックさせる
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="テキストが空です")
    if len(text) > TTS_MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text is too long. Keep it under {TTS_MAX_TEXT_LENGTH} characters.",
        )

    if config.TTS_PROVIDER == "edge":
        audio = await synthesize_speech_edge(text, req.voice)
    elif config.TTS_PROVIDER == "openai":
        audio = await synthesize_speech_openai(text, req.voice)
    elif config.TTS_PROVIDER == "browser":
        # フロントは503を受けてWeb Speech APIにフォールバックする
        raise HTTPException(
            status_code=503,
            detail="Server-side TTS is disabled. Use client-side speech synthesis.",
        )
    else:
        raise HTTPException(status_code=500, detail=f"未対応のTTSプロバイダー: {config.TTS_PROVIDER}")

    return Response(content=audio, media_type="audio/mpeg")


@app.get("/api/sessions")
async def list_sessions(limit: int = 20):
    """最近のセッション一覧を返す（弱点ダッシュボード用の土台）"""
    return JSONResponse({"sessions": db.get_recent_sessions(limit)})


@app.get("/api/corrections")
async def list_corrections(limit: int = 100):
    """全セッションの添削履歴を返す（弱点ダッシュボード用の土台）"""
    return JSONResponse({"corrections": db.get_all_corrections(limit)})


@app.get("/api/review/due")
async def review_due(limit: int = 10):
    """復習対象の添削リストを返す（未復習 or 復習期限が来たもの）
    error_type頻度の高い順 → 古い順で並ぶ（弱点から優先的に復習する）
    """
    limit = max(1, min(limit, 50))
    return JSONResponse({"reviews": db.get_due_reviews(limit)})


@app.post("/api/review/answer")
async def review_answer(req: ReviewAnswerRequest):
    """復習の解答を正誤判定してSRS状態を更新する
    - 判定はLLM不使用の正規化比較（高速・オフライン動作）
    - 完全一致で correct=True、類似度0.85以上なら close=True（惜しい）
    """
    correction = db.get_correction(req.correction_id)
    if correction is None:
        raise HTTPException(status_code=404, detail="Correction not found")

    result = judge_review_answer(req.answer_text, correction["corrected"] or "")
    # SRS状態を更新（正解なら間隔を延ばし、不正解なら翌日に再出題）
    srs = db.record_review_result(req.correction_id, result["correct"])

    return JSONResponse({
        "correct": result["correct"],
        "close": result["close"],
        "ratio": result["ratio"],
        "corrected": correction["corrected"],
        "explanation": correction["explanation"],
        "error_type": correction["error_type"],
        "srs": srs,
    })


@app.get("/api/stats")
async def get_stats():
    """弱点ダッシュボード用の統計を返す
    - total_corrections / total_turns / avg_score
    - error_stats: エラー種別ごとの件数と直近の例文（件数降順）
    - recent_scores: 直近の発音スコア推移（古い順）
    """
    overview = db.get_dashboard_stats()
    return JSONResponse({
        "total_corrections": overview["total_corrections"],
        "total_turns": overview["total_turns"],
        "avg_score": overview["avg_score"],
        "error_stats": db.get_error_stats(),
        "recent_scores": overview["recent_scores"],
    })


@app.get("/api/summary/{session_id}")
async def session_summary(session_id: str):
    """セッションサマリ（今日の学びレポート）を返す
    - 統計部分はルールベース（turn数 / 平均・最高・最低スコア / エラー内訳 / 学んだ表現）
    - 講評（feedback）はLLMを1回だけ呼んで生成。失敗時は null にして統計だけで200を返す
    """
    data = db.get_session_summary_data(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="このセッションにはまだ会話がありません")

    stats = build_session_summary_stats(data)

    # LLM講評（1回だけ呼び出し・失敗しても統計部分は返す）
    feedback = None
    try:
        raw = await get_llm_response(build_summary_feedback_messages(data))
        feedback = parse_summary_llm_json(raw)
    except Exception as e:
        # 接続不可・APIエラー等はすべて講評なし（null）で継続
        print(f"Summary feedback generation failed: {e}")
        feedback = None

    return JSONResponse({
        "session_id": session_id,
        "stats": stats,
        "feedback": feedback,
    })


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
        "tts_provider": config.TTS_PROVIDER,
        "tts_ready": is_tts_ready(),
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
