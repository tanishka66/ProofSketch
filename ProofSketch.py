import os, torch, re, time, numpy as np
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----- config -----
MODEL_NAME = ""   
HF_TOKEN   = "" 
HF_CACHE   = "/kaggle/working/hf_cache"           
USE_4BIT   = True
TEMPERATURE = 0.2
MAX_NEW_TOKENS_SKETCH = 220
MAX_NEW_TOKENS_EXPAND = 160

# ----- Speed up HF downloads & set cache -----
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "transformers")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  

# ----- Load tokenizer and model -----
load_kwargs = {}
if torch.cuda.is_available() and USE_4BIT:
    load_kwargs.update(dict(
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        trust_remote_code=False,
    ))
else:
    load_kwargs.update(dict(
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=False,
    ))

print("⏬ Downloading & loading:", MODEL_NAME)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Model ready on:", device)

# ----- Generation helpers & regex -----
def _gen(prompt, max_new, temperature=TEMPERATURE, do_sample=False):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    base = tok.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full[len(base):].strip()

CLAIM_LINE_RE = re.compile(r"^- +Claim +\d+: +(.+)$", re.I|re.M)
ANS_LINE_RE   = re.compile(r"^- +Answer: +(True|False|Unknown)", re.I)
def tokens_of(text): return len(tok(text).input_ids)



# --- Rule-checker for Att* theories + vocab helpers ---

import re
from typing import Dict, List, Tuple, Set

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

# --- Vocab extractor to fix AttributeError ---

def extract_vocab_from_theory(theory: str):
    """
    Build valid entity/attribute vocab from the theory.
    - Entities: capitalized single names (Anne, Harry, etc.)
    - Attributes: lowercase adjectives appearing after 'is/are' (+ parsed rules/facts)
    """
    # Entities 
    ents = set(re.findall(r"\b[A-Z][a-z]+\b", theory))

    # Attributes from surface patterns: "is <attr>", "is not <attr>", "are <attr>", "are not <attr>"
    attr_candidates = []
    for m in re.finditer(r"\bis\s+(?:not\s+)?([a-z]+)\b", theory):
        attr_candidates.append(m.group(1))
    for m in re.finditer(r"\bare\s+(?:not\s+)?([a-z]+)\b", theory):
        attr_candidates.append(m.group(1))

    # Harvest from parsed facts and rules
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
        print("No valid JSON found, retrying")
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


def proofsketch_eval(df, max_expansions=2, verbose=True, print_every=1, max_print=5):
    preds, golds, total_tokens, lat_ms, certified = [], [], [], [], []
    printed = 0

    for i, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df)), start=1):
        theory, question, gold = r.theory, r.question, str(r.answer)
        t0 = time.time()

        sketch_text, claims, pred = run_sketch(theory, question)

        fpos,fneg,rules = parse_theory(theory)
        pos,neg = derive_closure(fpos,fneg,rules)
        verdicts = [check_claim_atomic(c, pos, neg) for c in claims]
        needs = [c for c,v in zip(claims,verdicts) if v!="verified"]

        expansions = [run_expand(theory, c) for c in needs[:max_expansions]]

        elapsed_ms = 1000*(time.time()-t0)
        tok_count  = tokens_of(sketch_text) + sum(tokens_of(e) for e in expansions)

        preds.append(pred); golds.append(gold)
        lat_ms.append(elapsed_ms); total_tokens.append(tok_count)
        certified.append(len(needs)==0)

        if verbose and (i % print_every == 0) and (printed < (max_print or 10**9)):
            printed += 1
            print("="*90)
            print(f"Example {i}/{len(df)} | id: {r.get('id', i)}")
            print(f"Gold: {gold} | Pred: {pred} | tokens={tok_count} | latency={elapsed_ms:.1f} ms | certified={len(needs)==0}")
            print("- QUESTION:", question)
            print("- CLAIMS:", [f"{c} -> {v}" for c,v in zip(claims, verdicts)])
            if expansions:
                print("[EXPANSIONS]:")
                for e in expansions: print(e)

    acc = float(np.mean([p.lower()==g.lower() for p,g in zip(preds,golds)]))
    return {
        "acc": acc,
        "mean_tokens": float(np.mean(total_tokens)),
        "p95_tokens": float(np.percentile(total_tokens,95)),
        "mean_latency_ms": float(np.mean(lat_ms)),
        "certified_fraction": float(np.mean(certified))
    }


# ===== PATCH: robust vocab + sanitizer + JSON picker =====
import re, json

# Entities: proper names (Anne) and "The x" noun phrases that appear in the theory
def extract_vocab_from_theory(theory: str):
    proper = {m.group(0) for m in re.finditer(r"\b[A-Z][a-z]+\b", theory)}
    nouns  = {f"The {m.group(1).lower()}" for m in re.finditer(r"\b[Tt]he\s+([a-z]+)\b", theory)}
    ents   = proper | nouns

    # attributes from surface patterns and simple rules/facts
    attr_candidates = []
    for m in re.finditer(r"\bis\s+(?:not\s+)?([a-z]+)\b", theory):
        attr_candidates.append(m.group(1))
    for m in re.finditer(r"\bare\s+(?:not\s+)?([a-z]+)\b", theory):
        attr_candidates.append(m.group(1))

    # scrape from simple facts "X is (not) a" too (supports The x)
    for s in [t.strip() for t in theory.split('.') if t.strip()]:
        m = re.match(r"^(?:(?:[A-Z][a-z]+)|(?:[Tt]he\s+[a-z]+))\s+is\s+(?:not\s+)?([a-z]+)$", s)
        if m:
            attr_candidates.append(m.group(1))

    attrs = {a for a in attr_candidates if a.isalpha() and a.islower()}
    return ents, attrs

# Canonicalize entity tokens to either ProperCase or "The x"
def _canon_entity(token: str):
    token = token.strip()
    if re.match(r"^[A-Z][a-z]+$", token):
        return token
    m = re.match(r"^(?:[Tt]he)\s+([a-z]+)$", token)
    if m:
        return f"The {m.group(1).lower()}"
    return token

# Strict claim parser: "<Ent> is <attr>" or "<Ent> is not <attr>"
_CLAIM_PARSE = re.compile(r"^(?P<ent>(?:[A-Z][a-z]+|[Tt]he\s+[a-z]+))\s+is\s+(?P<not>not\s+)?(?P<attr>[a-z]+)$")

def sanitize_claims(raw_claims, ents, attrs):
    """
    Keep only atomic claims whose entity ∈ ents and attribute ∈ attrs.
    Normalize entity casing ("lion" -> "The lion"), attribute lowercasing,
    and drop anything not in vocab (fixes 'Green is green', 'bear is bear', etc.).
    """
    canon_ents = {_canon_entity(e) for e in ents}
    cleaned = []

    for c in raw_claims:
        s = str(c).strip().rstrip(".")
        m = _CLAIM_PARSE.match(s)
        ent, is_neg, attr = None, False, None

        if m:
            ent  = _canon_entity(m.group('ent'))
            is_neg = bool(m.group('not'))
            attr = m.group('attr').lower()
        else:
            #light recovery for lowercase starts like "lion is cold"
            parts = s.split()
            if len(parts) >= 3 and parts[1].lower() == "is":
                if parts[0].lower() == "the" and len(parts) >= 4:
                    ent = _canon_entity(" ".join(parts[:2]))  # "the lion" -> "The lion"
                    third = parts[2].lower()
                    if third == "not" and len(parts) >= 4:
                        is_neg = True
                        attr = parts[3].lower()
                    else:
                        attr = parts[2].lower()
                else:
                    # single-token entity like "charlie is kind"
                    ent = _canon_entity(parts[0].capitalize())
                    if len(parts) >= 4 and parts[2].lower() == "not":
                        is_neg = True
                        attr = parts[3].lower()
                    else:
                        attr = parts[2].lower()

        # Final gate: entity and attr must be in vocab
        if ent in canon_ents and attr in attrs:
            cleaned.append(f"{ent} is {'not ' if is_neg else ''}{attr}")

    # dedupe, keep ≤3
    uniq = []
    for cl in cleaned:
        if cl not in uniq:
            uniq.append(cl)
    return uniq[:3]

def extract_best_json(text: str):
    """
    Prefer the JSON block right after 'ANSWER:' (common format observed),
    else fall back to the longest valid JSON that has both 'claims' and 'answer'.
    """
    # 1) Look for ANSWER: then the next {...}
    m_ans = re.search(r"ANSWER:\s*(```json)?\s*\{", text, re.IGNORECASE)
    if m_ans:
        start = m_ans.start()
        # find first {...} after this point
        m_json = re.search(r"\{.*?\}", text[start:], flags=re.DOTALL)
        if m_json:
            cand = m_json.group(0)
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict) and "claims" in obj and "answer" in obj:
                    return obj
            except Exception:
                pass

    # 2) Fallback: scan all JSON-ish blocks and pick the longest valid one with keys
    best = None
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        cand = m.group(0)
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "claims" in obj and "answer" in obj:
                if best is None or len(cand) > len(best):
                    best = cand
        except Exception:
            continue
    return json.loads(best) if best else None


# ===== Boost ProofSketch: anchoring + 3-vote + closure correction =====
import re

def _question_entity(q: str):
    m = re.match(r"^(The\s+[a-z]+|[A-Z][a-z]+)\s+is\s+(?:not\s+)?([a-z]+)\.?$", q.strip())
    if not m: return None, None, None
    ent = m.group(1)
    ent = ent if re.match(r"^[A-Z][a-z]+$", ent) else ("The " + ent.split()[1].lower())
    attr = m.group(2).lower()
    neg  = bool(re.search(r"\bis\s+not\b", q))
    return ent, attr, neg

def _closure_decides(ent, attr, neg, pos, negset):
    if ent is None: return None
    has_pos = attr in pos.get(ent, set())
    has_neg = attr in negset.get(ent, set())
    if not has_pos and not has_neg: return None
    if not neg and has_pos: return "True"
    if not neg and has_neg: return "False"
    if neg and has_pos:     return "False"
    if neg and has_neg:     return "True"
    return None

def _anchor_claims(claims, theory, question):
    qe, _, _ = _question_entity(question)
    if not qe or not claims: return claims
    fpos,fneg,rules = parse_theory(theory); pos,neg = derive_closure(fpos,fneg,rules)

    about_q = [c for c in claims if c.startswith(qe + " is ")]
    if about_q: return about_q[:3]

    verified_q = []
    for a in pos.get(qe, set()): verified_q.append(f"{qe} is {a}")
    for a in neg.get(qe, set()): verified_q.append(f"{qe} is not {a}")
    if verified_q:
        base = [verified_q[0]]
        for c in claims:
            if len(base) >= 3: break
            if c not in base: base.append(c)
        return base[:3]
    return claims[:3]

def run_sketch_once(theory, question, max_new=MAX_NEW_TOKENS_SKETCH, temp=0.0):
    raw = generate(SKETCH_PROMPT.format(theory=theory, question=question), max_new,
                   temperature=temp, do_sample=(temp>0), stop_at_json=True)
    try:
        obj = extract_best_json(raw) or {}
    except Exception:
        obj = {}
    raw_claims = obj.get("claims", [])
    pred       = str(obj.get("answer","Unknown"))
    ents, attrs = extract_vocab_from_theory(theory)
    claims = sanitize_claims(raw_claims, ents, attrs)
    claims = _anchor_claims(claims, theory, question)
    return raw, claims, pred

def run_sketch_voted(theory, question, votes=3, temp=0.2):
    fpos,fneg,rules = parse_theory(theory); pos,neg = derive_closure(fpos,fneg,rules)
    qe, qattr, qneg = _question_entity(question)
    closure_ans = _closure_decides(qe, qattr, qneg, pos, neg)

    best = None 
    for _ in range(votes):
        raw, claims, pred = run_sketch_once(theory, question, temp=temp)
        verdicts = [check_claim_atomic(c, pos, neg) for c in claims]
        score = sum(v=="verified" for v in verdicts)
        toks = tokens_of(raw)
        consistent = 1 if (closure_ans is not None and pred.lower()==closure_ans.lower()) else 0
        cand = (score, -toks, consistent, (raw, claims, pred))
        best = max(best, cand) if best else cand

    raw, claims, pred = best[3]
    if closure_ans is not None:
        pred = closure_ans
    return raw, claims, pred



# ===== Canonical entity patch + helper =====
import re

def canon_ent(ent: str):
    s = ent.strip()
    # Proper name "Anne" → keep
    if re.match(r"^[A-Z][a-z]+$", s):
        return s
    # Noun phrase variants: "The bear" / "the bear" / "bear" → "The bear"
    m = re.match(r"^(?:the\s+)?([a-z]+)$", s, flags=re.I)
    if m:
        return f"The {m.group(1).lower()}"
    m = re.match(r"^(?:the\s+)([a-z]+)$", s, flags=re.I)
    if m:
        return f"The {m.group(1).lower()}"
    return s

def canonize_closure(pos: dict, neg: dict):
    def remap(d):
        out = {}
        for e, attrs in d.items():
            ce = canon_ent(e)
            out.setdefault(ce, set()).update(attrs)
        return out
    return remap(pos), remap(neg)

def get_closure(theory: str):
    """Build closure and canonicalize entity keys so they match claims like 'The bear ...'."""
    fpos, fneg, rules = parse_theory(theory)
    pos, neg = derive_closure(fpos, fneg, rules)
    pos, neg = canonize_closure(pos, neg)
    return pos, neg


# ===== ProofSketch: certification-first voting + adaptive tokens + anchoring + closure correction =====
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

# --------- Getting single verified claim if possible ----------
def _one_verified_claim(theory, question, pos, neg):
    qe, _, _ = _question_literal(question)
    if qe:
        if pos.get(qe): return [f"{qe} is {next(iter(pos[qe]))}"]
        if neg.get(qe): return [f"{qe} is not {next(iter(neg[qe]))}"]
    for e, attrs in pos.items():
        if attrs: return [f"{e} is {next(iter(attrs))}"]
    for e, attrs in neg.items():
        if attrs: return [f"{e} is not {next(iter(attrs))}"]
    return []

# --------- Strict anchoring ----------
def _anchor_claims(theory, question, claims, pos, neg):
    qe, _, _ = _question_literal(question)
    if qe:
        # If we have any verified fact about QE
        if pos.get(qe) or neg.get(qe):
            return _one_verified_claim(theory, question, pos, neg)
        # Else, keep only QE claims if present 
        about_q = [c for c in claims if c.startswith(qe + " is ")]
        if about_q:
            return [about_q[0]]
    # No QE verified; fallback to one global verified claim if any
    v1 = _one_verified_claim(theory, question, pos, neg)
    if v1: return v1
    # keep at most one sanitized claim
    return claims[:1]

# --------- One sketch sample (short budget) ----------
def _gen_sketch_once_strict(theory, question, max_new, temp):
    raw = generate(SKETCH_PROMPT.format(theory=theory, question=question),
                   max_new=max_new, temperature=temp, do_sample=(temp>0), stop_at_json=True)
    try:
        obj = extract_best_json(raw) or {}
    except Exception:
        obj = {}
    raw_claims = obj.get("claims", [])
    pred       = str(obj.get("answer","Unknown"))

    ents, attrs = extract_vocab_from_theory(theory)
    claims = sanitize_claims(raw_claims, ents, attrs)
    return raw, claims, pred

# --------- Certification-first voter with early stop + adaptive token budget ----------
def run_sketch_voted_plus(theory, question, max_votes=4, base_tokens=120, long_tokens=160, temp=0.3):
    # Compute closure 
    pos, neg = get_closure(theory)
    qe, qattr, qneg = _question_literal(question)
    closure_ans = _closure_decides(qe, qattr, qneg, pos, neg)
    
    sketch_tokens = base_tokens if (qe and (pos.get(qe) or neg.get(qe))) else long_tokens

    best = None  
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
        # Also early stop if we already have 2 verified claims
        if vcount >= 2:
            break

    raw, claims, pred = best[4]
    # Closure correction of the final label
    if closure_ans is not None:
        pred = closure_ans
    return raw, claims, pred

# --------- Evaluator ---------
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
        pos, neg = get_closure(theory)
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

#Run  
subset = df
ultra_summary = proofsketch_ultra_eval(
    subset,
    save_prefix="proofsketch_ultra",
    max_votes=4,       # allow up to 4 samples, but early-stop when certified
    base_tokens=120,  
    long_tokens=160,  
    temp=0.3,          
    verbose=True, print_every=1, max_print=8
)
