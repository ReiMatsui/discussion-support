"""LLMの要求パラメータと同期通信。セッション初期化から分離した共通処理。"""
from __future__ import annotations

import contextlib
import json

from ._constants import OPENAI_API

# 推論の強さは弱いほうから試す。短いJSON抽出に推論は要らないので最小を望むが、
# **どの値が使えるかはモデルの世代で変わる**。実際 gpt-5.4-mini は 'minimal' を
# 拒否し（使えるのは none/low/medium/high/xhigh）、既定モデルが上がった瞬間に
# LLM機能が全滅した（2026-07-29）。世代ごとの対応表を持つと、次にモデルが
# 上がったとき同じことが起きる。だから**弾かれたら次を試して覚える**。
_EFFORT_ORDER = ("minimal", "none", "low")
# モデル名（小文字）-> そのモデルで通った値。セッション内だけ持つ。
_EFFORT: dict[str, str] = {}


def _build_chat_params(model: str, prompt: str, *, max_out: int,
                       temperature: float, schema_name: str | None = None,
                       schema: dict | None = None) -> dict:
    """Chat Completions のリクエストパラメータを構築する（モデル系統別）.

    gpt-5系/o系は推論モデルで、temperature指定不可・max_tokensは
    max_completion_tokensに改名。さらに推論トークンが出力枠を消費するため、
    枠が小さいと本文(JSON)が空のまま返る → 静かに失敗する（Fix 9の根因）。
    対策として reasoning_effort を抑え、出力枠を十分に確保する。
    """
    name = model.lower()
    params: dict = {"model": model,
                    "messages": [{"role": "user", "content": prompt}]}
    if schema_name and schema:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
    if name.startswith("gpt-5"):
        params["reasoning_effort"] = _EFFORT.get(name, _EFFORT_ORDER[0])
        params["max_completion_tokens"] = max_out
    elif name.startswith(("o1", "o3", "o4")):
        params["reasoning_effort"] = "low"       # o系は minimal 非対応
        params["max_completion_tokens"] = max_out
    else:
        params["temperature"] = temperature
        params["max_tokens"] = max_out
    return params


def _next_effort(params: dict, detail: str) -> str | None:
    """`reasoning_effort` が拒否されたら、次に試す値を返す（無ければ None）.

    弾かれた値より弱い順の次を選び、モデル名に対して覚える。以後の呼び出しは
    `_build_chat_params` がその値で組む（1セッション1回だけ余分に叩く）。
    """
    if "reasoning_effort" not in detail:
        return None      # 別の理由の 400（モデル名違い等）はここで扱わない
    cur = params.get("reasoning_effort")
    if cur not in _EFFORT_ORDER:
        return None
    nxt = _EFFORT_ORDER[_EFFORT_ORDER.index(cur) + 1:]
    if not nxt:
        return None
    _EFFORT[str(params.get("model", "")).lower()] = nxt[0]
    return nxt[0]


def _post_chat_json(params: dict, api_key: str, *, timeout: int, label: str,
                    _retried: bool = False, raise_auth_errors: bool = False):
    """Chat Completions を叩き、本文をJSONとして返す。失敗時はNone（理由をログ）."""
    import urllib.error
    import urllib.request
    body = json.dumps(params).encode()
    req = urllib.request.Request(OPENAI_API, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        text = (resp["choices"][0]["message"].get("content") or "").strip()
        if not text:
            # 空応答（推論で出力枠を使い切った等）— 静かに失敗させず可視化する
            finish = resp["choices"][0].get("finish_reason")
            print(f"# [{label}] 空応答（finish_reason={finish}）。"
                  f"max_completion_tokens不足の可能性", flush=True)
            return None
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except urllib.error.HTTPError as e:
        # **本文を必ず出す**。OpenAI は 400 の理由（未知のモデル名、その
        # モデルでは使えないパラメータ等）を本文に書いてくるので、ここを
        # 捨てると「400 Bad Request」だけが延々と流れて原因が分からない
        # （2026-07-29 に実会話で発生し、LLM機能が全滅しているのに何が
        # 悪いのか特定できなかった）。
        detail = ""
        with contextlib.suppress(Exception):
            detail = e.read().decode("utf-8", "replace")[:600]
        nxt = None if _retried else _next_effort(params, detail)
        if nxt is not None:
            print(f"# [{label}] reasoning_effort='{params['reasoning_effort']}' は"
                  f"{params.get('model')} で使えないため '{nxt}' で再試行します",
                  flush=True)
            return _post_chat_json({**params, "reasoning_effort": nxt}, api_key,
                                   timeout=timeout, label=label, _retried=True,
                                   raise_auth_errors=raise_auth_errors)
        # 認証エラーの本文にはキーの一部が返る場合があるため、端末・保存ログへ出さない。
        log_detail = "認証またはアクセス権を確認してください" if e.code in (401, 403) else detail or e.reason
        print(f"# [{label}] APIエラー {e.code}: {log_detail}"
              f"（model={params.get('model')}）", flush=True)
        if raise_auth_errors and e.code in (401, 403):
            raise
        return None
    except Exception as e:
        print(f"# [{label}] API/解析エラー: {type(e).__name__}: {e}", flush=True)
        return None


