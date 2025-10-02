NO_COT_PROMPT = """You are a precise reasoner. Read the THEORY and answer the QUESTION with one word: True, False, or Unknown.
Respond in STRICT JSON: {{\"answer\": \"<True|False|Unknown>\"}} and NOTHING else.

THEORY:
{theory}

QUESTION:
{question}
"""

SHORT_COT_PROMPT = """You are a concise reasoner. Provide at most 3 short lines of reasoning, then the final answer.
Return STRICT JSON:
{{\"rationale\": [\"...\", \"...\"], \"answer\": \"<True|False|Unknown>\"}} and NOTHING else.

THEORY:
{theory}

QUESTION:
{question}
"""

LONG_COT_PROMPT = """You are a careful reasoner. Think step by step for up to 10 short lines, then give the final answer.
Return STRICT JSON:
{{\"rationale\": [\"...\", \"...\", \"...\"], \"answer\": \"<True|False|Unknown>\"}} and NOTHING else.

THEORY:
{theory}

QUESTION:
{question}
"""

# === Baseline evaluator (saves logs + summary) ===
import re, json, time, numpy as np, pandas as pd
from tqdm import tqdm

def _extract_answer_only(text):
    cand = None
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if "answer" in obj: cand = obj
        except: pass
    return (str(cand.get("answer","Unknown")), json.dumps(cand)) if cand else ("Unknown", text)

def baseline_eval_logged(df, prompt_tmpl, save_prefix, max_new=220):
    rows=[]; preds=[]; golds=[]; toks=[]; lats=[]
    for _, r in tqdm(df.iterrows(), total=len(df)):
        t0=time.time()
        raw = generate(prompt_tmpl.format(theory=r.theory, question=r.question), max_new, temperature=0.0)
        lats.append(1000*(time.time()-t0))
        pred, kept = _extract_answer_only(raw)
        preds.append(pred); golds.append(str(r.answer))
        toks.append(tokens_of(kept))
        rows.append({"id": r.get("id",""), "question": r.question, "gold": str(r.answer),
                     "pred": pred, "kept": kept, "tokens": toks[-1], "latency_ms": lats[-1]})
    acc = float(np.mean([p.lower()==g.lower() for p,g in zip(preds,golds)]))
    summary = {"acc": acc, "mean_tokens": float(np.mean(toks)), "p95_tokens": float(np.percentile(toks,95)),
               "mean_latency_ms": float(np.mean(lats)), "n_examples": int(len(df))}
    pd.DataFrame(rows).to_csv(f"{save_prefix}_logs.csv", index=False)
    with open(f"{save_prefix}_summary.json","w") as f: json.dump(summary,f,indent=2)
    print(f"Saved {save_prefix}_logs.csv and {save_prefix}_summary.json")
    print("Summary:", summary)
    return summary

subset = df  # same slice you used for ProofSketch

nocot   = baseline_eval_logged(subset, NO_COT_PROMPT,    "baseline_nocot")
shortct = baseline_eval_logged(subset, SHORT_COT_PROMPT, "baseline_shortcot")
longct  = baseline_eval_logged(subset, LONG_COT_PROMPT,  "baseline_longcot")
