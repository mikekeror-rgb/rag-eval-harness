# rag-eval-harness

A [Claude Code](https://claude.com/claude-code) skill that adds a repeatable evaluation harness to an existing RAG app: a golden question set, retrieval metrics (hit@k, MRR), judge-scored faithfulness and relevancy, citation and refusal checks, and a CI quality gate. It also checks the metrics themselves, because the most expensive eval bug is a metric that can't fail.

## Case study: what auditing a real RAG eval turned up

The checks in this skill come from evaluating the [Advocate Remuneration Assistant](https://github.com/mikekeror-rgb/advocate_renumeration_assistant_v1), a RAG assistant for Kenyan advocates' fees, scored on 236 pre-registered questions with a Qwen2.5-7B model. Its headline was 87.7% correct. Auditing the evaluation found:

- **Two router bugs.** Digits inside case citations (`2026KEHC13078`) were read as a Kshs 13,078 fee, and a calculation guard fired on any question that mentioned calculating. Six of 236 questions got a canned calculator reply instead of an answer. Fixed, the same model scored 213 instead of 206 in a paired run.
- **A retrieval metric that passed by construction on 72% of questions** (171 of 236). It counted any chunk of the right type as a hit, so it reported 98.3% for ruling questions while the gold passage itself was retrieved for 78.6%.
- **A judge that rewarded wrong refusals.** The faithfulness judge gave a perfect 5/5 to 7 of the base model's 10 wrong "Not covered in the provided documents" answers. A refusal is always "supported by the context".
- **A citation check stricter than the thing it checked.** Exact matching flagged real, retrieved chunks as fabricated when the model fixed the corpus's misspelling ("Renumeration" → "Remuneration"). On the 62-question set, 6 answers contained an invented citation, not the 11 reported.

With the metrics fixed, a retrieval-only harness (no model, no judge, about 30 seconds per run) screened candidate retrieval changes. Re-ranking was the one worth a full run: gold-passage recall went from 80.0% to 90.0%, accuracy from 89.4% to 93.6% (+4.2 points, 95% interval +0.4 to +8.2), and wrong refusals from 10 to 1. These later gains are exploratory, because the changes were chosen by looking at the test set's failures.

Those problems were found with the project's own harness. This skill packages the checks that caught them (Steps 6 and 7 of `SKILL.md`), so they run as part of the default workflow.

## What the skill does

When you ask Claude Code to evaluate, benchmark or regression-test a RAG app, it:

1. Reads your code to find the retriever, the generator and the citation format.
2. Writes a thin adapter around your real pipeline (never a re-implementation).
3. Helps you build a golden set: answers written from the source documents, gold chunk ids found by hand, out-of-scope questions included.
4. Sets up a judge model (Groq free tier, a local Ollama model, or OpenAI), never the same small model as the generator.
5. Runs the harness:

| Metric | What it tells you | Needs a judge? |
|---|---|---|
| Hit@k, MRR | Did the gold chunk make the top k, and how high? | No |
| Faithfulness | Is every claim supported by retrieved context? | Yes |
| Answer relevancy | Does the answer address the question? | Yes |
| Citation validity | Do `[N]` markers point to chunks that exist? | No |
| Refusal rate | Were out-of-scope questions refused? | No |
| False-refusal rate | Were answerable questions refused? | No |

6. Checks every metric can fail, moves when it should, and doesn't reward refusals, and measures the noise floor with an identical rerun.
7. Compares changes as paired runs, with per-question flips and intervals, one change at a time.
8. Optionally adds a GitHub Actions gate: retrieval-only metrics on every PR, the judged run on `main`.

Each run writes `evals/results.md` (overall and per question type) and `evals/results.jsonl` (per question, with the retrieved ids and text), so you can re-score a run without regenerating answers.

## Install

Copy the folder into a skills directory Claude Code reads:

```bash
git clone https://github.com/mikekeror-rgb/rag-eval-harness.git
cp -R rag-eval-harness ~/.claude/skills/rag-eval-harness        # all projects
# or: cp -R rag-eval-harness <your-project>/.claude/skills/      # one project
```

Then, in a RAG project, ask Claude Code to "evaluate my RAG app".

To run the script on its own, put `scripts/run_eval.py` in `evals/` next to an `evals/rag_adapter.py` that exposes `answer(question) -> {"answer", "contexts", "context_ids"}`:

```bash
pip install deepeval openai
python evals/run_eval.py --retrieval-only --min-hit 0.75            # fast, no judge
python evals/run_eval.py --min-faithfulness 0.8 --max-false-refusal 0.05
```

## Files

| Path | Purpose |
|---|---|
| `SKILL.md` | The workflow Claude Code follows |
| `scripts/run_eval.py` | Metrics, results table, per-question log, quality gate |
| `templates/golden_set.jsonl` | Golden-set format: lookup, calculation, multi-hop, out-of-scope |
| `templates/rag-eval.yml` | GitHub Actions workflow |

## License

MIT. Built by [Mike Keror](https://github.com/mikekeror-rgb).
