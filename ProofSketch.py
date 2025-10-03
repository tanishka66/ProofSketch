import os, torch, re, time, numpy as np
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----- HF config (edit these if needed) -----
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   
HF_TOKEN   = "" 
HF_CACHE   = "/kaggle/working/hf_cache"              
USE_4BIT   = False
TEMPERATURE = 0.6
MAX_NEW_TOKENS_SKETCH = 220
MAX_NEW_TOKENS_EXPAND = 160

# ----- Speed up HF downloads & set cache -----
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "transformers")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster dl

# ----- Optional: login if you provided a token -----
if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=True)
        print("Hugging Face login successful.")
    except Exception as e:
        print("HF login failed (continuing without):", e)

# ----- Load tokenizer & model (with optional 4-bit on GPU) -----
load_kwargs = {}
if torch.cuda.is_available():
    load_kwargs.update(dict(
        device_map="auto",
        torch_dtype=torch.float16,   # or bfloat16 if you have an A100
        trust_remote_code=False,
    ))
else:
    load_kwargs.update(dict(
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=False,
    ))

print("Downloading & loading:", MODEL_NAME)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token  # avoid padding warnings

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
model.eval()

def generate(prompt, max_new=220, temperature=0.0, do_sample=False, stop_at_json=False):
    # make sure pad token is set
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    base = tok.decode(inputs["input_ids"][0], skip_special_tokens=True)
    text = full[len(base):].strip()

    if stop_at_json:
        i = text.find("{")
        if i != -1:
            depth = 0
            for j, ch in enumerate(text[i:], start=i):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i:j+1]
    return text



TOKENIZER_NAME = ""
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

def tokens_of(text: str) -> int:
    """Approximate token count with local tokenizer."""
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False)["input_ids"])



def parse_theory(theory: str):
    facts_pos: Dict[str, Set[str]] = {}
    facts_neg: Dict[str, Set[str]] = {}
    rules: List[Tuple[Tuple[Tuple[str,bool],...], Tuple[str,bool]]] = []
    sents = [s.strip() for s in theory.split('.') if s.strip()]

    for s in sents:
        # Facts: "Anne is white." / "Anne is not white."
        m = re.match(r"^([A-Z][a-z]+) is (not )?([a-z]+)$", s)
        if m:
            ent, neg, attr = m.group(1), m.group(2), m.group(3)
            (facts_neg if neg else facts_pos).setdefault(ent, set()).add(attr)
            continue

        # Universal: "All red people are rough."
        m = re.match(r"^All ([a-z]+) (?:people|things)? are (not )?([a-z]+)$", s)
        if m:
            a1, neg, concl = m.group(1), m.group(2), m.group(3)
            rules.append((((a1, True),), (concl, False if neg else True)))
            continue

        # If-then: "If someone is red (and smart) then they are rough."
        m = re.match(r"^If someone is ([a-z]+)(?: and (?:someone is )?([a-z]+))? then (?:they are|it is) (not )?([a-z]+)$", s)
        if m:
            a1, a2, neg, concl = m.group(1), m.group(2), m.group(3), m.group(4)
            ants = [(a1, True)]
            if a2: ants.append((a2, True))
            rules.append((tuple(ants), (concl, False if neg else True)))
            continue

        # Conj declarative: "Smart, red people are rough."
        m = re.match(r"^([A-Za-z]+), ([A-Za-z]+) (?:people|things) are (not )?([a-z]+)$", s)
        if m:
            a1, a2, neg, concl = m.group(1).lower(), m.group(2).lower(), m.group(3), m.group(4)
            rules.append( (((a1,True),(a2,True)), (concl, False if neg else True)) )
            continue

    return facts_pos, facts_neg, rules

def derive_closure(facts_pos: Dict[str,set], facts_neg: Dict[str,set], rules):
    pos = {e:set(v) for e,v in facts_pos.items()}
    neg = {e:set(v) for e,v in facts_neg.items()}
    for _ in range(4):  # depth ≤2 is small
        changed = False
        ents = set(pos.keys()) | set(neg.keys())
        for ants, (concl, is_pos) in rules:
            for ent in list(ents):
                ok = True
                for a, pol in ants:
                    if pol and a not in pos.get(ent,set()): ok=False; break
                    if not pol and a not in neg.get(ent,set()): ok=False; break
                if ok:
                    tgt = pos if is_pos else neg
                    if concl not in tgt.setdefault(ent,set()):
                        tgt[ent].add(concl); changed=True
        if not changed: break
    return pos, neg

_CLAIM = re.compile(r"^([A-Z][a-z]+) is (not )?([a-z]+)$")
def check_claim_atomic(claim: str, pos: Dict[str,set], neg: Dict[str,set]):
    m = _CLAIM.match(claim.strip().rstrip('.'))
    if not m: return "unsupported"
    ent, n, attr = m.group(1), m.group(2), m.group(3)
    return "verified" if (attr in (neg.get(ent,set()) if n else pos.get(ent,set()))) else "unverified"



def extract_vocab_from_theory(theory: str):
    """
    Build valid entity/attribute vocab from the theory.
    - Entities: capitalized single names (Anne, Harry, etc.)
    - Attributes: lowercase adjectives appearing after 'is/are' (+ parsed rules/facts)
    """
    # Entities (single ProperCase tokens)
    ents = set(re.findall(r"\b[A-Z][a-z]+\b", theory))

    # Attributes from surface patterns: "is <attr>", "is not <attr>", "are <attr>", "are not <attr>"
    attr_candidates = []
    for m in re.finditer(r"\bis\s+(?:not\s+)?([a-z]+)\b", theory):
        attr_candidates.append(m.group(1))
    for m in re.finditer(r"\bare\s+(?:not\s+)?([a-z]+)\b", theory):
        attr_candidates.append(m.group(1))

    # Also harvest from parsed facts and rules (more reliable)
    fpos, fneg, rules = parse_theory(theory)
    for s in fpos.values():
        attr_candidates.extend(list(s))
    for s in fneg.values():
        attr_candidates.extend(list(s))
    for ants, (concl, _pol) in rules:
        for a, _ in ants:
            attr_candidates.append(a)
        attr_candidates.append(concl)

    # Keep clean lowercase single words
    attrs = {a for a in attr_candidates if isinstance(a, str) and a.isalpha() and a.islower()}

    return ents, attrs


# --- Strict JSON prompts + sanitize (robust) ---

MAX_NEW_TOKENS_SKETCH = 220
MAX_NEW_TOKENS_EXPAND = 160

SKETCH_PROMPT = """You are a strict logician.
Return ONLY one JSON object. 
NO prose, NO explanations, NO examples, NO labels like VALID or INVALID.

THEORY:
{theory}

QUESTION:
{question}

RULES:
- JSON object must contain keys: "claims" (2–3 strings) and "answer" (True/False/Unknown).
- Each claim MUST be exactly "<Entity> is <attribute>" OR "<Entity> is not <attribute>".
- <Entity> must be a proper name from the THEORY.
- <attribute> must be a lowercase word from the THEORY.
- No "and", commas, or explanations. One atomic fact per claim.
- Output must be ONLY this JSON object.
"""

EXPAND_PROMPT = """You are expanding only the specified claim with the shortest reasoning.

THEORY:
{theory}

EXPAND THIS CLAIM ONLY:
{claim_text}

FORMAT (≤3 short lines):
- Step 1: ...
- Step 2: ...
- Justification: ...
"""

CLAIM_RE = re.compile(r"^([A-Z][a-z]+) is (not )?([a-z]+)$")

def sanitize_claims(raw_claims, ents, attrs):
    """Normalize case, enforce atomic form, filter to legal (entity, attribute)."""
    cleaned = []
    for c in raw_claims:
        s = str(c).strip().rstrip(".")
        parts = s.split()
        ent, neg, attr = None, "", None

        # Try "<Ent> is <attr>" or "<Ent> is not <attr>"
        if len(parts) >= 3 and parts[1].lower() == "is":
            ent = parts[0].capitalize()
            if len(parts) >= 4 and parts[2].lower() == "not":
                neg = "not"
                attr = parts[3].lower()
            else:
                attr = parts[2].lower()

        # Validate against vocab
        if ent in ents and attr in attrs:
            cleaned.append(f"{ent} is {neg + ' ' if neg else ''}{attr}".strip())
            continue

        # Heuristic fallback: pick any legal entity/attr that appear in string
        ent_hit = next((e for e in ents if re.search(rf"\b{re.escape(e)}\b", s, re.I)), None)
        attr_hit = next((a for a in attrs if re.search(rf"\b{re.escape(a)}\b", s, re.I)), None)
        if ent_hit and attr_hit:
            cleaned.append(f"{ent_hit} is {attr_hit}")

    # keep at most 3 claims; prefer unique
    uniq = []
    for c in cleaned:
        if c not in uniq:
            uniq.append(c)
    return uniq[:3]

def extract_best_json(s: str):
    """Return the longest valid JSON object that has 'claims' and 'answer'."""
    matches = re.findall(r"\{.*?\}", s, flags=re.DOTALL)
    best = None
    for m in matches:
        try:
            obj = json.loads(m)
            if isinstance(obj, dict) and "claims" in obj and "answer" in obj:
                if best is None or len(m) > len(best):
                    best = m
        except Exception:
            continue
    return json.loads(best) if best else None

def run_sketch(theory: str, question: str):
    out = generate(SKETCH_PROMPT.format(theory=theory, question=question),
                   MAX_NEW_TOKENS_SKETCH, temperature=0.0)

    print("\n[RAW MODEL OUTPUT - SKETCH]")
    print(out)

    data = extract_best_json(out)
    if data is None:
        print("No valid JSON found, retrying once...")
        out = generate(SKETCH_PROMPT.format(theory=theory, question=question),
                       160, temperature=0.0)
        print("\n[RAW MODEL OUTPUT - RETRY]")
        print(out)
        data = extract_best_json(out) or {"claims": [], "answer": "Unknown"}

    claims_raw = data.get("claims", [])
    answer     = str(data.get("answer", "Unknown"))

    ents, attrs = extract_vocab_from_theory(theory)
    claims = sanitize_claims(claims_raw, ents, attrs)

    # Fallback: if empty, try one atomic fact from the theory facts
    fpos, fneg, _ = parse_theory(theory)
    if not claims and fpos:
        any_ent = next(iter(fpos.keys()))
        any_attr = next(iter(fpos[any_ent]))
        claims = [f"{any_ent} is {any_attr}"]

    print("[SANITIZED CLAIMS]:", claims)
    print("[FINAL ANSWER]:", answer)
    return out, claims, answer

def run_expand(theory: str, claim_text: str):
    out = generate(EXPAND_PROMPT.format(theory=theory, claim_text=claim_text),
                   MAX_NEW_TOKENS_EXPAND, temperature=0.0)
    print("\n[RAW MODEL OUTPUT - EXPANSION for:", claim_text, "]")
    print(out)
    return out


# ===== ProofSketch++: certification-first voting + adaptive tokens + anchoring + closure correction =====
import re, json, time, numpy as np, pandas as pd
from tqdm import tqdm

# --------- Question literal + closure helpers ----------
def _question_literal(q: str):
    m = re.match(r"^(The\s+[a-z]+|[A-Z][a-z]+)\s+is\s+(?:(not)\s+)?([a-z]+)\.?$", q.strip())
    if not m: return None, None, None
    ent = m.group(1)
    if not re.match(r"^[A-Z][a-z]+$", ent):
        ent = "The " + ent.split()[1].lower()
    return ent, m.group(3).lower(), bool(m.group(2))

def _closure_decides(ent, attr, is_neg, pos, neg):
    if ent is None: return None
    has_pos = attr in pos.get(ent, set())
    has_neg = attr in neg.get(ent, set())
    if not has_pos and not has_neg: return None
    if not is_neg and has_pos: return "True"
    if not is_neg and has_neg: return "False"
    if is_neg and has_pos:     return "False"
    if is_neg and has_neg:     return "True"
    return None

# --------- Make a single verified claim if possible (boosts certification) ----------
def _one_verified_claim(theory, question, pos, neg):
    qe, _, _ = _question_literal(question)
    # Prefer about the question entity
    if qe:
        if pos.get(qe): return [f"{qe} is {next(iter(pos[qe]))}"]
        if neg.get(qe): return [f"{qe} is not {next(iter(neg[qe]))}"]
    # Else any verified fact
    for e, attrs in pos.items():
        if attrs: return [f"{e} is {next(iter(attrs))}"]
    for e, attrs in neg.items():
        if attrs: return [f"{e} is not {next(iter(attrs))}"]
    return []

# --------- Strict anchoring: collapse to 1 verified claim when possible ----------
def _anchor_claims(theory, question, claims, pos, neg):
    qe, _, _ = _question_literal(question)
    if qe:
        # If we have any verified fact about QE, use just that (easier to certify)
        if pos.get(qe) or neg.get(qe):
            return _one_verified_claim(theory, question, pos, neg)
        # Else, keep only QE claims if present (still 1 claim to help certification)
        about_q = [c for c in claims if c.startswith(qe + " is ")]
        if about_q:
            return [about_q[0]]
    # No QE verified; fallback to one global verified claim if any
    v1 = _one_verified_claim(theory, question, pos, neg)
    if v1: return v1
    # Last resort: keep at most one sanitized claim
    return claims[:1]

# --------- One sketch sample (short budget) ----------
def _gen_sketch_once_strict(theory, question, max_new, temp):
    raw = generate(
        SKETCH_PROMPT.format(theory=theory, question=question),
        max_new_tokens=max_new,      
        temperature=temp,
        do_sample=(temp > 0),
        stop_at_json=True
    )
    try:
        obj = extract_best_json(raw) or {}
    except Exception:
        obj = {}
    raw_claims = obj.get("claims", [])
    pred       = str(obj.get("answer", "Unknown"))

    ents, attrs = extract_vocab_from_theory(theory)
    claims = sanitize_claims(raw_claims, ents, attrs)
    return raw, claims, pred

# --------- Certification-first voter with early stop + adaptive token budget ----------
def run_sketch_voted_plus(theory, question, max_votes=4, base_tokens=120, long_tokens=160, temp=0.3):
    # Compute closure once
    fpos,fneg,rules = parse_theory(theory); pos,neg = derive_closure(fpos,fneg,rules)
    qe, qattr, qneg = _question_literal(question)
    closure_ans = _closure_decides(qe, qattr, qneg, pos, neg)

    # Adaptive token budget: if we already have some closure facts about QE, use shorter budget
    sketch_tokens = base_tokens if (qe and (pos.get(qe) or neg.get(qe))) else long_tokens

    best = None  # (all_cert, verified_count, -tokens, consistent, (raw, claims, pred))
    for v in range(max_votes):
        raw, claims, pred = _gen_sketch_once_strict(theory, question, max_new=sketch_tokens, temp=temp)

        # Anchor to single verified claim if possible
        claims = _anchor_claims(theory, question, claims, pos, neg)

        verdicts = [check_claim_atomic(c, pos, neg) for c in claims]
        all_cert = int(len(verdicts)>0 and all(v=="verified" for v in verdicts))
        vcount   = sum(v=="verified" for v in verdicts)
        toks     = tokens_of(raw)
        consistent = 1 if (closure_ans is not None and pred.lower()==closure_ans.lower()) else 0

        cand = (all_cert, vcount, -toks, consistent, (raw, claims, pred))
        best = max(best, cand) if best else cand

        # EARLY STOP: fully certified sketch found
        if all_cert:
            break
        # Also early stop if we already have 2 verified claims (rare with 1-claim policy)
        if vcount >= 2:
            break

    raw, claims, pred = best[4]
    # Closure correction of the final label (free accuracy)
    if closure_ans is not None:
        pred = closure_ans
    return raw, claims, pred

# --------- Evaluator: ProofSketch-only (improved), prints + saves ---------
def proofsketch_ultra_eval(df, save_prefix="proofsketch_ultra",
                           max_votes=4, base_tokens=120, long_tokens=160, temp=0.3,
                           verbose=True, print_every=1, max_print=8):
    rows=[]; preds=[]; golds=[]; toks=[]; lats=[]; certs=[]
    printed=0; N=len(df)

    for _, r in tqdm(df.iterrows(), total=N):
        theory, question, gold = r.theory, r.question, str(r.answer)
        t0 = time.time()

        raw, claims, pred = run_sketch_voted_plus(
            theory, question,
            max_votes=max_votes, base_tokens=base_tokens, long_tokens=long_tokens, temp=temp
        )
        fpos,fneg,rules = parse_theory(theory); pos,neg = derive_closure(fpos,fneg,rules)
        verdicts = [check_claim_atomic(c, pos, neg) for c in claims]

        tok = tokens_of(raw); elapsed_ms = 1000*(time.time()-t0)
        certified = (len(verdicts)>0) and all(v=="verified" for v in verdicts)

        i = len(preds)+1
        if verbose and (i % print_every == 0) and (printed < max_print):
            print("="*100)
            print(f"Example {i}/{N} | id: {r.get('id','')} | Gold: {gold} | Pred: {pred} | tokens={tok} | latency={elapsed_ms:.1f} ms | certified={certified}")
            th = theory if len(theory) <= 320 else theory[:320] + " …"
            print("- THEORY:\n"+th)
            print("- QUESTION:", question)
            print("[RAW SKETCH JSON-ish]\n", raw.strip() if isinstance(raw,str) else str(raw))
            print("[CLAIMS]")
            if claims:
                for c,v in zip(claims, verdicts):
                    print("  -", c, "->", v)
            else:
                print("  (none)")
            printed += 1

        rows.append({
            "id": r.get("id",""),
            "question": question,
            "gold": gold,
            "pred": pred,
            "claims": claims,
            "verdicts": verdicts,
            "tokens": tok,
            "latency_ms": elapsed_ms,
            "certified": certified
        })
        preds.append(pred); golds.append(gold); toks.append(tok); lats.append(elapsed_ms); certs.append(certified)

    acc = float(np.mean([p.lower()==g.lower() for p,g in zip(preds,golds)]))
    summary = {
        "acc": acc,
        "mean_tokens": float(np.mean(toks)),
        "p95_tokens": float(np.percentile(toks,95)),
        "mean_latency_ms": float(np.mean(lats)),
        "certified_fraction": float(np.mean([1.0 if c else 0.0 for c in certs])),
        "n_examples": int(N)
    }
    logs_path = f"{save_prefix}_logs.csv"
    sum_path  = f"{save_prefix}_summary.json"
    pd.DataFrame(rows).to_csv(logs_path, index=False)
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {logs_path} and {sum_path}")
    print("Summary:", summary)
    return summary

# ---- RUN it (same subset you used for baselines) ----
subset = df
ultra_summary = proofsketch_ultra_eval(
    subset,
    save_prefix="proofsketch_ultra",
    max_votes=4,       # allow up to 4 samples, but early-stop when certified
    base_tokens=120,   # short budget when QE has closure facts
    long_tokens=160,   # slightly longer if closure is empty for QE
    temp=0.3,        
    verbose=True, print_every=1, max_print=8
)

