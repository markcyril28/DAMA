"""Laya stdio worker: answers Dama move questions as JSON lines.

Runs under the interpreter that has laya installed (``python -I -u``), never the
Dama one: it imports the standard library and laya only, never ``dama``.
Importing this module has no side effects; all process setup happens in main().

Protocol (one JSON object per line): the worker reports ``{"event": "ready"}``
or ``{"event": "fatal"}`` once, then answers ``ping`` and ``choose`` requests
read from stdin until EOF or ``{"op": "shutdown"}``. Replies go to a duplicate
of the original stdout; fd 1 itself is redirected to stderr because laya prints
its warnings there.
"""

import argparse
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

# laya.common.build_sequence keeps at most this many tokens of each option.
MAX_OPTION_TOKENS = 48
# build_sequence cuts every option when fewer than this many head tokens remain.
MIN_OPTION_ROOM = 16

_WARMUP_REQUEST = {
    "id": 0,
    "op": "choose",
    "state": "warm-up",
    "instructions": "Warm-up question: pick either option.",
    "options": ["yes", "no"],
}


def _ids(tok, text: str) -> list[int]:
    """Token ids of text exactly as build_sequence tokenizes it (no special tokens)."""
    return list(tok(text, add_special_tokens=False)["input_ids"])


def check_budget(tok, state, q: dict, max_len: int, base_head_max_len: int, *,
                 build_sequence, render_options, serialize_state) -> tuple[int, list[str]]:
    """Return (head_max_len to use, problems) so build_sequence keeps every input token."""
    mask = tok.mask_token
    full_head = _ids(tok, "%s question: %s" % (q["t"], str(q["ins"]).replace(mask, " ")))
    rendered = render_options(q)
    opt_full = [_ids(tok, " " + text.replace(mask, " ")) for text in rendered]

    problems = []
    too_long = set()
    for i, ids in enumerate(opt_full):
        if len(ids) > MAX_OPTION_TOKENS:
            too_long.add(i)
            problems.append("option %d %r is %d tokens; laya keeps only the first %d"
                            % (i, rendered[i], len(ids), MAX_OPTION_TOKENS))

    options_len = sum(1 + len(ids) for ids in opt_full)
    hml = max(int(base_head_max_len), options_len + max(MIN_OPTION_ROOM, len(full_head)))

    # Verify on laya's real output rather than trusting the arithmetic above.
    seq, markers = build_sequence(tok, state, q, max_len, hml)
    seq = list(seq)
    markers = list(markers)
    head_end = 1 + len(full_head)
    if seq[1:head_end] != full_head or len(seq) <= head_end or seq[head_end] != tok.sep_token_id:
        problems.append("the %d-token instructions do not fit intact (max_len=%d)"
                        % (len(full_head), max_len))
    if len(markers) != len(opt_full):
        problems.append("only %d of %d options fit in max_len=%d (head needs %d tokens)"
                        % (len(markers), len(opt_full), max_len, 3 + len(full_head) + options_len))
    else:
        for i, (pos, ids) in enumerate(zip(markers, opt_full)):
            if i in too_long:
                continue
            if (pos >= len(seq) or seq[pos] != tok.mask_token_id
                    or seq[pos + 1:pos + 1 + len(ids)] != ids):
                problems.append("option %d %r is not intact in the model input" % (i, rendered[i]))

    st_full = _ids(tok, serialize_state(state).replace(mask, " "))
    expected_len = 4 + len(full_head) + options_len + len(st_full)
    state_intact = (len(seq) >= len(st_full) + 1 and seq[-1] == tok.sep_token_id
                    and seq[len(seq) - len(st_full) - 1:-1] == st_full)
    if len(seq) > max_len or not state_intact or (not too_long and len(seq) != expected_len):
        room = max(0, max_len - (3 + len(full_head) + options_len) - 1)
        problems.append("the state is %d tokens but only %d fit in max_len=%d after the %d-token head"
                        % (len(st_full), room, max_len, 3 + len(full_head) + options_len))
    return hml, problems


def _error(rid: Any, code: str, message: str) -> dict:
    """Failure reply for one request."""
    return {"id": rid, "ok": False, "code": code, "error": message}


def _validate_choose(req: dict) -> str:
    """Return why a choose request is malformed, or '' when it is usable."""
    options = req.get("options")
    if not isinstance(options, list) or not options:
        return "options must be a non-empty list of strings"
    if not all(isinstance(o, str) and o for o in options):
        return "every option must be a non-empty string"
    if len(set(options)) != len(options):
        return "options must be unique (laya collapses duplicate labels)"
    if not isinstance(req.get("state"), (str, dict)):
        return "state must be a string or an object"
    if not isinstance(req.get("instructions"), str):
        return "instructions must be a string"
    return ""


def serve_choose(agent, req: dict, device_arg: str, fns) -> dict:
    """Answer one choose request; fns carries laya's build_sequence/render_options/serialize_state."""
    rid = req.get("id")
    device_before = agent.device.type
    try:
        bad = _validate_choose(req)
        if bad:
            return _error(rid, "error", bad)
        options = list(req["options"])
        state = req["state"]
        qdef = {"type": "choice", "instructions": req["instructions"], "criteria": list(options)}
        q = agent._to_internal(qdef)
        max_len = agent.cfg.get("max_len", 512)
        base = agent.cfg.get("head_max_len", 192)
        hml, problems = check_budget(agent.tok, state, q, max_len, base,
                                     build_sequence=fns.build_sequence,
                                     render_options=fns.render_options,
                                     serialize_state=fns.serialize_state)
        if problems:
            return _error(rid, "budget", "; ".join(problems))

        # system_one reads head_max_len from agent.cfg on every call.
        had_key = "head_max_len" in agent.cfg
        original = agent.cfg.get("head_max_len")
        agent.cfg["head_max_len"] = hml
        started = time.perf_counter()
        try:
            result = agent.predict(state, {"move": qdef})
        finally:
            if had_key:
                agent.cfg["head_max_len"] = original
            else:
                agent.cfg.pop("head_max_len", None)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        answer = result["answers"]["move"]
        probabilities = [float(answer["probabilities"][o]) for o in options]
        index = options.index(answer["choice"])
        device = agent.device.type
        if device_arg == "cuda" and device != "cuda":
            return _error(rid, "device", "cuda was requested but laya moved the model to %s" % device)
        return {"id": rid, "ok": True, "probabilities": probabilities, "choice_index": index,
                "confidence": float(answer["confidence"]), "device": device,
                "elapsed_ms": elapsed_ms, "head_max_len": hml}
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        device_after = agent.device.type
        if device_after != device_before or (device_arg == "cuda" and device_after != "cuda"):
            # laya switches agent.device to cpu before it moves the model; when that
            # move or the retry fails, the model is left split across devices and
            # every later call fails, so only a fresh worker can serve again.
            return _error(rid, "device", "laya is on %s after a failed call on %s: %r"
                          % (device_after, device_before, exc))
        if device_before == "cuda" and isinstance(exc, RuntimeError) and "cuda" in str(exc).lower():
            # An asynchronous kernel fault surfaces at laya's first host sync, outside
            # its fallback: agent.device still reads cuda but the CUDA context is dead.
            return _error(rid, "device", "CUDA failure on %s: %r" % (device_before, exc))
        return _error(rid, "error", repr(exc))


def _resolve_root(model: str, subfolder: str) -> str:
    """Checkpoint root directory: a local directory as is, else the hub snapshot."""
    local = os.path.expanduser(model)
    if os.path.isdir(local):
        return os.path.abspath(local)
    if model.startswith(("/", "./", "../", "~")) or os.path.isabs(model):
        raise FileNotFoundError("Local model path not found: %r" % model)
    from huggingface_hub import constants as hf_constants
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    # Same patterns as laya.Agent, so only the selected checkpoint is fetched.
    prefix = "%s/" % subfolder if subfolder else ""
    patterns = [prefix + name for name in
                ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")]
    try:
        return snapshot_download(model, allow_patterns=patterns)
    except LocalEntryNotFoundError as exc:  # IncompleteSnapshotError is a subclass
        if not hf_constants.HF_HUB_OFFLINE:
            raise
        # The hub's own advice (set HF_HUB_OFFLINE=0) cannot work: the client
        # forces it on while the offline setting is on.
        raise FileNotFoundError(
            "Laya weights for %r%s are not available from the offline Hugging Face cache %s "
            "(%s). Point the HF cache dir at the cache that holds them (Settings > Laya AI > HF "
            "cache dir, or --hf-home), set Model to a local checkpoint directory such as a "
            "snapshot folder that has them, or set 'offline: false' under ai.laya in "
            "settings.yaml to allow a download"
            % (model, " subfolder %r" % subfolder if subfolder else "",
               hf_constants.HF_HUB_CACHE, type(exc).__name__)) from exc


def _parser() -> argparse.ArgumentParser:
    """Command line of the worker."""
    parser = argparse.ArgumentParser(description="Laya stdio worker for Dama.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--subfolder", default="")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--expected-laya-version", required=True,
                        help="comma-separated laya versions this protocol was verified against")
    return parser


def _startup(args) -> tuple:
    """Load the agent, warm it up and return (agent, laya functions, ready event)."""
    started = time.perf_counter()
    expected = [v.strip() for v in args.expected_laya_version.split(",") if v.strip()]
    import laya
    from laya import common as laya_common

    version = getattr(laya, "__version__", None)
    if version not in expected:
        raise RuntimeError("laya %s is installed but this bridge supports only %s"
                           % (version, ", ".join(expected) or "(none)"))
    fns = SimpleNamespace(build_sequence=laya_common.build_sequence,
                          render_options=laya_common.render_options,
                          serialize_state=laya_common.serialize_state)

    root = _resolve_root(args.model, args.subfolder)
    device = None if args.device == "auto" else args.device
    agent = laya.load(root, device=device, subfolder=args.subfolder or None)
    if args.device == "cuda" and agent.device.type != "cuda":
        raise RuntimeError("cuda was requested but laya loaded the model on %s" % agent.device.type)
    load_sec = time.perf_counter() - started

    warm = serve_choose(agent, dict(_WARMUP_REQUEST), args.device, fns)
    if not warm.get("ok"):
        raise RuntimeError("warm-up choice failed (%s): %s" % (warm.get("code"), warm.get("error")))

    import torch

    root = os.path.normpath(root)
    parent = os.path.basename(os.path.dirname(root))
    ready = {
        "event": "ready",
        "pid": os.getpid(),
        "device": agent.device.type,
        "laya_version": version,
        "model": args.model,
        "subfolder": args.subfolder,
        "snapshot_path": root,
        "revision": os.path.basename(root) if parent == "snapshots" else None,
        "head_max_len": agent.cfg.get("head_max_len", 192),
        "max_len": agent.cfg.get("max_len", 512),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_device": torch.cuda.get_device_name(agent.device) if agent.device.type == "cuda" else None,
        "load_sec": round(load_sec, 3),
        "warmup_ms": round(float(warm.get("elapsed_ms", 0.0)), 3),
    }
    return agent, fns, ready


def main(argv=None) -> int:
    """Serve requests until EOF or shutdown; returns the process exit code."""
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    sys.stdout.flush()
    os.dup2(2, 1)  # laya prints warnings to stdout; keep them off the protocol stream

    def send(obj: dict) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    try:
        try:
            args = _parser().parse_args(argv)
        except SystemExit as exc:
            send({"event": "fatal", "error": "bad worker arguments (argparse exit %s)" % exc.code})
            return 2
        agent, fns, ready = _startup(args)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({"event": "fatal", "error": repr(exc)})
        return 1
    send(ready)

    stdin = os.fdopen(os.dup(0), "r", encoding="utf-8", errors="replace")
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as exc:
            send(_error(None, "error", "request is not JSON: %r" % exc))
            continue
        if not isinstance(req, dict):
            send(_error(None, "error", "request must be a JSON object"))
            continue
        op, rid = req.get("op"), req.get("id")
        if op == "shutdown":
            break
        if op == "ping":
            send({"id": rid, "ok": True, "device": agent.device.type})
        elif op == "choose":
            send(serve_choose(agent, req, args.device, fns))
        else:
            send(_error(rid, "error", "unknown op %r" % (op,)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
