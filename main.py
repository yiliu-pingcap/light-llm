# main.py
import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# --------- sampling helpers ---------
def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, _ = torch.topk(logits, k)
    cutoff = v[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)

    # mask tokens whose cumulative prob exceeds p (keep at least 1 token)
    mask = cumprobs > p
    mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    # unsort back
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return out


@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    # logits: (vocab,)
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / max(temperature, 1e-8)
    logits = _top_k_logits(logits, top_k)
    logits = _top_p_logits(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# --------- model loading ---------
def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        return ckpt
    # sometimes people save raw state_dict
    return {"state_dict": ckpt}


def try_import_model():
    """
    Tries a few common module/class names. Adjust as needed for your repo.
    Expected: a torch.nn.Module class that can be constructed with **model_args (optional)
              and returns logits when called:
                  logits = model(input_ids)  # (B, T, V)
              or:
                  logits, *rest = model(input_ids)
    """
    candidates = [
        ("model", "Model"),
        ("model", "Transformer"),
        ("net", "Model"),
        ("net", "Transformer"),
        ("llm", "Model"),
    ]
    last_err = None
    for module_name, class_name in candidates:
        try:
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)
            return cls
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        "Could not import a model class. Tried: "
        + ", ".join([f"{m}.{c}" for m, c in candidates])
        + f"\nLast error: {last_err}"
    )


def build_model(ckpt: dict, device: torch.device, dtype: torch.dtype):
    ModelCls = try_import_model()

    model_args = ckpt.get("model_args", None) or ckpt.get("config", None) or {}
    state = ckpt.get("state_dict", None)

    if state is None:
        # maybe weights are at top-level
        # (heuristic: if keys look like parameter tensors)
        tensor_keys = [k for k, v in ckpt.items() if torch.is_tensor(v)]
        if tensor_keys:
            state = {k: ckpt[k] for k in tensor_keys}
        else:
            raise ValueError("Checkpoint has no 'state_dict' and doesn't look like a raw state_dict.")

    # Some repos store state under "model" or "module"
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "module" in state and isinstance(state["module"], dict):
        state = state["module"]

    try:
        model = ModelCls(**model_args)
    except TypeError:
        # If your model ctor takes no kwargs, just instantiate it.
        model = ModelCls()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    model.eval()
    model.to(device=device)
    # move params/buffers to dtype
    for p in model.parameters():
        p.data = p.data.to(dtype=dtype)
    for b in model.buffers():
        if torch.is_tensor(b):
            b.data = b.data.to(dtype=dtype)
    return model


# --------- generation ---------
@torch.no_grad()
def forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns logits (B, T, V). Supports model returning logits or (logits, ...).
    """
    out = model(input_ids)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    stream: bool = True,
) -> str:
    # encode prompt
    input_ids = tokenizer.encode(prompt)
    if not isinstance(input_ids, (list, tuple)):
        raise TypeError("tokenizer.encode(prompt) must return a list/tuple of token ids.")
    ids = torch.tensor(input_ids, dtype=torch.long, device=device)[None, :]  # (1, T)

    # print prompt immediately if streaming (optional)
    if stream:
        sys.stdout.write("")
        sys.stdout.flush()

    decoded_so_far = prompt
    start_len = ids.size(1)

    for _ in range(max_new_tokens):
        logits = forward_logits(model, ids)  # (1, T, V)
        next_logits = logits[0, -1, :]       # (V,)

        next_id = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # append
        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
        ids = torch.cat([ids, next_tok], dim=1)

        if eos_id is not None and next_id == eos_id:
            break

        if stream:
            # decode only the newly generated part (best-effort)
            new_text = tokenizer.decode(ids[0, start_len:].tolist())
            # print delta
            delta = new_text[len(decoded_so_far) - len(prompt) :] if new_text.startswith(decoded_so_far[len(prompt):]) else new_text
            sys.stdout.write(delta)
            sys.stdout.flush()

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return tokenizer.decode(ids[0].tolist())


# --------- CLI ---------
def parse_args():
    ap = argparse.ArgumentParser(description="Lightweight LLM runner (expects tokenizer.py + your model code).")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt/.pth).")
    ap.add_argument("--prompt", type=str, default="", help="Prompt text (ignored if --interactive).")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--interactive", action="store_true", help="REPL mode.")
    ap.add_argument("--no_stream", action="store_true", help="Disable streaming token output.")
    return ap.parse_args()


def pick_dtype(dtype_flag: str, device: str) -> torch.dtype:
    if dtype_flag == "fp32":
        return torch.float32
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "bf16":
        return torch.bfloat16
    # auto
    if device.startswith("cuda"):
        # bf16 if supported, else fp16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype, args.device)

    # import tokenizer
    from tokenizer import Tokenizer  # you said you already have tokenizer.py
    tokenizer = Tokenizer()

    # try to discover eos id if tokenizer exposes it
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    model = build_model(ckpt, device=device, dtype=dtype)

    stream = not args.no_stream

    if args.interactive:
        print("Interactive mode. Type 'exit' or Ctrl-D to quit.\n")
        while True:
            try:
                prompt = input("> ")
            except EOFError:
                print()
                break
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            if not prompt.strip():
                continue
            t0 = time.time()
            _ = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_id=eos_id,
                device=device,
                stream=stream,
            )
            if not stream:
                print(_)
            dt = time.time() - t0
            print(f"[time] {dt:.2f}s\n")
    else:
        if not args.prompt:
            print("Provide --prompt or use --interactive.", file=sys.stderr)
            sys.exit(2)
        out = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=eos_id,
            device=device,
            stream=stream,
        )
        if not stream:
            print(out)


if __name__ == "__main__":
    main()
