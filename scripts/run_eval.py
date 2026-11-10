"""RAG eval harness runner.

Usage:
  python evals/run_eval.py --golden evals/golden_set.jsonl --k 5
  python evals/run_eval.py --retrieval-only --min-hit 0.8      # fast CI mode, no judge
  python evals/run_eval.py --max-false-refusal 0.05            # also gate on refusing answerable questions

Expects evals/rag_adapter.py exposing answer(question) -> {answer, contexts, context_ids}.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

REFUSAL_PATTERNS = [
    r"\bi (do not|don't) (know|have)\b", r"\bnot (covered|mentioned|found) in\b",
    r"\bno information\b", r"\bcannot (answer|find)\b", r"\boutside (the|my) scope\b",
]


def load_golden(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_scores(retrieved_ids, relevant_ids, k):
    top = retrieved_ids[:k]
    if not relevant_ids:
        return None, None
    hit = float(any(r in relevant_ids for r in top))
    rr = 0.0
    for rank, rid in enumerate(top, start=1):
        if rid in relevant_ids:
            rr = 1.0 / rank
            break
    return hit, rr


def citation_valid(answer, n_contexts):
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    if not cited:
        return None  # app does not cite; metric not applicable
    return float(all(1 <= c <= n_contexts for c in cited))


def is_refusal(answer):
    a = answer.lower()
    return any(re.search(p, a) for p in REFUSAL_PATTERNS)


def build_judge():
    """OpenAI-compatible judge (Groq, Ollama, OpenAI) wrapped for DeepEval."""
    from deepeval.models import DeepEvalBaseLLM
    from openai import OpenAI

    client = OpenAI(base_url=os.getenv("JUDGE_BASE_URL") or None,
                    api_key=os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY"))
    model_name = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

    class Judge(DeepEvalBaseLLM):
        def load_model(self):
            return client

        def generate(self, prompt: str) -> str:
            r = client.chat.completions.create(
                model=model_name, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self):
            return model_name

    return Judge()


def judge_scores(item, out, judge):
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    tc = LLMTestCase(input=item["question"], actual_output=out["answer"],
                     expected_output=item.get("expected_answer"),
                     retrieval_context=out["contexts"])
    scores = {}
    for name, metric in [("faithfulness", FaithfulnessMetric(model=judge, async_mode=False)),
                         ("relevancy", AnswerRelevancyMetric(model=judge, async_mode=False))]:
        for attempt in range(2):  # small judges often emit malformed JSON
            try:
                metric.measure(tc)
                scores[name] = metric.score
                break
            except Exception as e:  # record, don't crash the whole run
                if attempt == 1:
                    scores[name] = None
                    scores.setdefault("judge_errors", []).append(f"{name}: {type(e).__name__}")
    return scores


def mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="evals/golden_set.jsonl")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--out", default="evals/results.md")
    p.add_argument("--min-hit", type=float)
    p.add_argument("--min-faithfulness", type=float)
    p.add_argument("--max-false-refusal", type=float)
    args = p.parse_args()

    from rag_adapter import answer

    golden = load_golden(args.golden)
    judge = None if args.retrieval_only else build_judge()
    rows, log = [], []
    for item in golden:
        row = {"id": item["id"], "type": item.get("type", "lookup")}
        try:
            out = answer(item["question"])
        except Exception as e:  # rate limit, timeout, app bug: record it and keep going
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rows.append(row)
            log.append({**row, "question": item["question"]})
            print(f"{row['id']}: ERROR {row['error'][:120]}", flush=True)
            write_outputs(rows, log, args, len(golden))
            continue
        hit, rr = retrieval_scores(out["context_ids"], item.get("relevant_ids", []), args.k)
        row.update({"hit": hit, "mrr": rr, "citation": citation_valid(out["answer"], len(out["contexts"]))})
        if item.get("type") == "out_of_scope":
            row["refused"] = float(is_refusal(out["answer"]))
        else:
            # Faithfulness judges score "I don't know" as fully supported, so track
            # refusals of answerable questions separately or they look like wins.
            row["false_refusal"] = float(is_refusal(out["answer"]))
        if judge and item.get("type") != "out_of_scope":
            row.update(judge_scores(item, out, judge))
        rows.append(row)
        log.append({**row, "question": item["question"], "answer": out["answer"],
                    "context_ids": out["context_ids"], "contexts": out["contexts"]})
        print(f"{row['id']}: hit={hit} mrr={rr} faith={row.get('faithfulness')}", flush=True)
        write_outputs(rows, log, args, len(golden))  # a crash later still leaves results on disk

    overall, lines = write_outputs(rows, log, args, len(golden))
    print("\n".join(lines))

    failed = []
    errored = [r["id"] for r in rows if "error" in r]
    if errored:  # averages skip errored questions, so a run with errors must not pass
        failed.append(f"{len(errored)} question(s) errored: {', '.join(errored)}")
    if args.min_hit is not None and (overall["hit"] or 0) < args.min_hit:
        failed.append(f"hit {overall['hit']} < {args.min_hit}")
    if args.min_faithfulness is not None and (overall["faithfulness"] or 0) < args.min_faithfulness:
        failed.append(f"faithfulness {overall['faithfulness']} < {args.min_faithfulness}")
    if args.max_false_refusal is not None and (overall["false_refusal"] or 0) > args.max_false_refusal:
        failed.append(f"false_refusal {overall['false_refusal']} > {args.max_false_refusal}")
    if failed:
        print("QUALITY GATE FAILED: " + "; ".join(failed))
        sys.exit(1)


METRICS = ["hit", "mrr", "faithfulness", "relevancy", "citation", "refused", "false_refusal"]


def write_outputs(rows, log, args, total):
    """Write results.md (summary) and results.jsonl (per question) for the rows so far."""
    overall = {m: mean(r.get(m) for r in rows) for m in METRICS}
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    status = f"{len(rows)} questions" if len(rows) == total else f"PARTIAL: {len(rows)} of {total} questions so far"
    lines = [f"# RAG eval results\n\n{status}, k={args.k}\n",
             "| Slice | " + " | ".join(METRICS) + " |",
             "|---|" + "---|" * len(METRICS),
             "| **all** | " + " | ".join(str(overall[m]) for m in METRICS) + " |"]
    for t, rs in sorted(by_type.items()):
        lines.append(f"| {t} ({len(rs)}) | " +
                     " | ".join(str(mean(r.get(m) for r in rs)) for m in METRICS) + " |")
    app_errors = [r for r in rows if "error" in r]
    if app_errors:
        lines.append(f"\nErrored, excluded from the averages ({len(app_errors)}): "
                     + "; ".join(f"{r['id']} {r['error'][:80]}" for r in app_errors))
    judge_errors = [(r["id"], e) for r in rows for e in r.get("judge_errors", [])]
    if judge_errors:
        lines.append(f"\nJudge failures ({len(judge_errors)}): " + ", ".join(f"{i} {e}" for i, e in judge_errors))
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Per-question log: re-score later (fixed metric, other judge) without regenerating answers.
    with open(Path(args.out).with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for entry in log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return overall, lines


if __name__ == "__main__":
    main()
