"""A stand-in for the SGLang server, for testing the generation pipeline without a GPU.

Purpose : lets src/generate.py and scripts/deltaai/serve_and_generate.sh be run end to end
          on a laptop. It accepts the same launch arguments as `python -m sglang.launch_server`
          (unknown ones are ignored) and serves the endpoints generate.py uses, with SGLang's
          response shape:
            GET  /health                 -> 200 once "loaded" (after --fake-load-s seconds)
            GET  /get_server_info        -> {"version": "fake-sglang", "model_path": ..., ...}
            GET  /get_model_info         -> {"model_path": ...}
            POST /generate {input_ids, sampling_params, return_logprob}
                 -> {"text", "meta_info": {"finish_reason": {"type": "stop"|"length", ...},
                     "prompt_tokens", "completion_tokens", "output_token_logprobs": [[lp, id, None], ...]}}
          It does NOT run a model: the "answer" is synthetic text, so no GPQA content is involved.
Behaviour: think-mode prompts (ending in "<think>\\n") get reasoning + </think> + an answer field;
          default-mode prompts get visible reasoning + an answer field. Output longer than
          max_new_tokens is cut with finish_reason "length". With --fake-error-every N, every
          Nth /generate request returns HTTP 500 (to test error logging).
Usage   : python -m tests.fake_sglang --model-path Qwen/Qwen3.5-2B --port 30000 [--fake-error-every 7]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"tok": None, "args": None, "ready_at": 0.0, "n_requests": 0, "lock": threading.Lock()}


def fake_completion(input_ids: list[int], tok) -> str:
    """Deterministic synthetic output for a prompt (depends only on the input ids)."""
    h = int(hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(), 16)
    letter = "ABCD"[h % 4]
    n_steps = 3 + h % 40  # varying length, so truncation can be tested with a small cap
    prompt = tok.decode(input_ids[-8:], skip_special_tokens=False)
    reasoning = " ".join(f"Step {i}: compare the options." for i in range(n_steps))
    if prompt.endswith("<think>\n"):  # think mode: the open block is in the prompt
        return f"{reasoning}\n</think>\n\n{{\"answer\": \"{letter}\"}}"
    return f"{reasoning}\n\n{{\"answer\": \"{letter}\"}}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep test output quiet
        pass

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        a = STATE["args"]
        if time.time() < STATE["ready_at"]:
            return self._send(503, {"error": "loading"})
        if self.path == "/health":
            return self._send(200, {})
        if self.path in ("/get_server_info", "/server_info"):
            return self._send(200, {"version": "fake-sglang", "model_path": a.model_path, "revision": a.revision,
                                    "dtype": a.dtype, "random_seed": a.random_seed})
        if self.path == "/get_model_info":
            return self._send(200, {"model_path": a.model_path, "is_generation": True})
        return self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, {"error": f"no route {self.path}"})
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with STATE["lock"]:
            STATE["n_requests"] += 1
            n = STATE["n_requests"]
        every = STATE["args"].fake_error_every
        if every and n % every == 0:
            return self._send(500, {"error": "fake internal server error"})
        tok = STATE["tok"]
        ids = tok(fake_completion(req["input_ids"], tok), add_special_tokens=False)["input_ids"]
        cap = req["sampling_params"]["max_new_tokens"]
        if len(ids) > cap:
            ids, finish = ids[:cap], {"type": "length", "length": cap}
        else:
            ids, finish = ids + [tok.eos_token_id], {"type": "stop", "matched": tok.eos_token_id}
        time.sleep(STATE["args"].fake_latency_s)
        meta = {"id": f"fake-{n}", "finish_reason": finish, "prompt_tokens": len(req["input_ids"]),
                "completion_tokens": len(ids), "cached_tokens": 0, "e2e_latency": STATE["args"].fake_latency_s}
        if req.get("return_logprob"):
            meta["output_token_logprobs"] = [[-0.5, t, None] for t in ids]
        text = tok.decode(ids[:-1] if finish["type"] == "stop" else ids, skip_special_tokens=False)
        return self._send(200, {"text": text, "meta_info": meta})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--fake-error-every", type=int, default=0)
    ap.add_argument("--fake-latency-s", type=float, default=0.01)
    ap.add_argument("--fake-load-s", type=float, default=1.0, help="seconds before /health turns 200")
    args, _unknown = ap.parse_known_args(argv)  # ignore real SGLang flags we don't model
    from transformers import AutoTokenizer
    STATE["tok"] = AutoTokenizer.from_pretrained(args.model_path, revision=args.revision)
    STATE["args"] = args
    STATE["ready_at"] = time.time() + args.fake_load_s
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"fake-sglang listening on {args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
