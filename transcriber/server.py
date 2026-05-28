#!/usr/bin/env python3
"""
Live Transcriber - OpenAI Realtime API proxy server.

Browser <--WebSocket--> this server <--WebSocket--> OpenAI Realtime API
                                          ^
                                          |
                                  OPENAI_API_KEY (server side only)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# --- Dependency bootstrap ---------------------------------------------------
def _ensure_deps():
    try:
        import aiohttp  # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    print("[setup] aiohttp が無いのでインストールします...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "aiohttp"]
    )

_ensure_deps()

import aiohttp
from aiohttp import web, WSMsgType, ClientWebSocketResponse


HERE = Path(__file__).parent.resolve()


# --- API key loading --------------------------------------------------------
def load_api_key() -> str:
    # Priority: env var > .env in this folder > ~/.openai_api_key
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()

    env_file = HERE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    return v

    home_key = Path.home() / ".openai_api_key"
    if home_key.exists():
        v = home_key.read_text().strip()
        if v:
            return v

    print(
        "\n[error] OPENAI_API_KEY が見つかりません。次のいずれかで設定してください:\n"
        f"  1) このフォルダに .env を作成: OPENAI_API_KEY=sk-...\n"
        f"  2) ~/.openai_api_key にキーだけを保存\n"
        "  3) 環境変数 OPENAI_API_KEY を設定して再実行\n",
        file=sys.stderr,
    )
    sys.exit(1)


API_KEY = load_api_key()

# Available transcription models (cheapest -> most accurate):
#   gpt-4o-mini-transcribe, gpt-4o-transcribe, whisper-1
DEFAULT_MODEL = os.environ.get("TRANSCRIBE_MODEL", "gpt-4o-transcribe")

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


# --- WebSocket relay --------------------------------------------------------
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws_client = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await ws_client.prepare(request)
    print("[+] Browser connected")

    http_session = aiohttp.ClientSession()
    ws_openai: ClientWebSocketResponse | None = None
    try:
        try:
            ws_openai = await http_session.ws_connect(
                OPENAI_REALTIME_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "OpenAI-Beta": "realtime=v1",
                },
                max_msg_size=16 * 1024 * 1024,
                heartbeat=30,
            )
            print("[+] OpenAI Realtime connected")
        except Exception as e:
            print(f"[!] OpenAI connection failed: {e}")
            await ws_client.send_json(
                {
                    "type": "proxy.error",
                    "error": {"message": f"OpenAIへの接続に失敗: {e}"},
                }
            )
            await ws_client.close()
            return ws_client

        # Send the initial transcription session config now (server-side)
        # so the client doesn't need to know all the parameters.
        await ws_openai.send_json(
            {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": DEFAULT_MODEL,
                        # language omitted => auto detect (English + Japanese mix OK)
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                    "input_audio_noise_reduction": {"type": "near_field"},
                },
            }
        )

        async def client_to_openai() -> None:
            try:
                async for msg in ws_client:
                    if msg.type == WSMsgType.TEXT:
                        await ws_openai.send_str(msg.data)
                    elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                        break
            except Exception as e:
                print(f"[!] client->openai: {e}")

        async def openai_to_client() -> None:
            try:
                async for msg in ws_openai:
                    if msg.type == WSMsgType.TEXT:
                        await ws_client.send_str(msg.data)
                    elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                        break
            except Exception as e:
                print(f"[!] openai->client: {e}")

        await asyncio.gather(client_to_openai(), openai_to_client())
    finally:
        if ws_openai is not None:
            await ws_openai.close()
        await http_session.close()
        if not ws_client.closed:
            await ws_client.close()
        print("[-] Connection closed")
    return ws_client


# --- Optional: server-side completion endpoint for translate/summarize ------
async def complete_handler(request: web.Request) -> web.Response:
    """Lightweight chat-completion proxy used for translation/summary."""
    body = await request.json()
    prompt: str = body.get("prompt", "")
    model: str = body.get("model", "gpt-4o-mini")
    if not prompt:
        return web.json_response({"error": "prompt required"}, status=400)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                return web.json_response(data, status=resp.status)
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return web.json_response({"text": text})


# --- Static files -----------------------------------------------------------
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(HERE / "index.html", headers=NO_CACHE_HEADERS)


# --- App boot ---------------------------------------------------------------
def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/complete", complete_handler)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    print("=" * 56)
    print("  Live Transcriber (OpenAI Realtime API)")
    print(f"  Model: {DEFAULT_MODEL}")
    print(f"  URL:   http://localhost:{port}")
    print("=" * 56)
    web.run_app(make_app(), host="127.0.0.1", port=port, print=None)
