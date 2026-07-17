# AI English Conversation Tutor

AI English Conversation Tutor is a local-first English speaking practice app built with FastAPI and a static browser UI. It supports grammar correction, short conversation replies, IPA pronunciation hints, and speech transcription.

The default setup uses local providers where possible:

- LLM: Ollama
- STT: faster-whisper
- Optional cloud LLM/STT: OpenAI or Anthropic via environment variables

## Features

- Chat-style English conversation practice
- Strict grammar correction with JSON-structured feedback
- Natural expression suggestions
- IPA pronunciation support
- Pronunciation scoring based on recognized English text
- Local Ollama support for private LLM inference
- Local faster-whisper support for speech-to-text
- Server-side text-to-speech with edge-tts (natural neural voices), with OpenAI TTS and browser speech synthesis as alternatives
- Optional OpenAI and Anthropic provider support

## Safety

API keys are not stored in source code. Configure secrets through environment variables or a local `.env` file.

Never commit:

- `.env`
- API keys
- `key.pem`
- `cert.pem`
- generated certificates or private keys

The repository includes `.env.example` as a safe template.

## Requirements

- Python 3.10+
- Optional: Ollama for local LLM inference
- Optional: faster-whisper local model download
- Optional: OpenAI or Anthropic API key for cloud providers

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Default local configuration:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma2:latest
STT_PROVIDER=faster_whisper
FASTER_WHISPER_MODEL=base.en
TTS_PROVIDER=edge
EDGE_TTS_VOICE=en-US-JennyNeural
EDGE_TTS_RATE=-10%
```

Text-to-speech options for `TTS_PROVIDER`:

- `edge` (default): server-side neural voices via edge-tts
- `openai`: OpenAI TTS (requires `OPENAI_API_KEY`)
- `browser`: client-side Web Speech API only

If server-side TTS is unavailable, the UI automatically falls back to the browser's built-in speech synthesis.

Optional cloud configuration:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=replace-with-your-openai-key
OPENAI_MODEL=gpt-4o-mini
```

or:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=replace-with-your-anthropic-key
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

## Run

Windows:

```bat
start.bat
```

macOS/Linux:

```bash
./start.sh
```

Or run directly:

```bash
python server.py
```

The server uses local HTTPS certificates (`cert.pem` and `key.pem`). If they do not exist, the app generates self-signed certificates for local development.

## Project Structure

```text
.
├── config.py              # Provider and app configuration
├── server.py              # FastAPI backend
├── static/
│   ├── index.html         # Browser UI
│   └── bee.svg            # UI asset
├── requirements.txt
├── .env.example
├── start.bat
└── start.sh
```

## Notes

This app is intended as a local learning tool and portfolio project. For public deployment, configure production-grade HTTPS, authentication, CORS, and secret management.
