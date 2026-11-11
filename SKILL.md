---
name: rag-eval-harness
description: Add a repeatable evaluation harness to an existing RAG app - golden Q&A set, retrieval metrics (hit@k, MRR), DeepEval faithfulness/relevancy with a free judge model, citation checks, false-refusal rate, sanity checks that catch metrics which cannot fail, paired before/after comparisons, a results table and a CI quality gate. Use when someone asks to evaluate, benchmark, test or regression-check a RAG / retrieval-augmented chatbot.
---

# RAG Eval Harness

Turns "the chatbot seems fine" into numbers you can put in a README and fail a build on.
Works with any RAG app that can return its answer **and** the chunks it retrieved.

## Output you will produce

```
evals/
  golden_set.jsonl      # 20-50 questions with expected answers + relevant chunk ids
  rag_adapter.py        # thin wrapper: question -> answer + retrieved chunks
  run_eval.py           # copied from this skill's scripts/
  results.md            # metrics table, regenerated on every run
  results.jsonl         # per-question answers, retrieved ids and scores, for re-scoring
  negative_controls.jsonl  # clean/broken pairs that test the metrics and judge themselves
  controls.md           # which control pairs behaved as expected
.github/workflows/rag-eval.yml   # optional CI gate
```

## Step 1 - Inspect the app (do not skip)

Read the codebase and find:
1. **The retriever call** - the function that returns chunks for a query. Note `k`, and whether chunks carry a stable id (doc name + chunk index, a hash, etc.).
2. **The generator call** - the LLM call that takes question + chunks and returns the answer.
3. **Citation format** - does the answer cite sources like `[1]`, `[2]`? If yes, the citation check in Step 5 applies.

If chunks have no stable id, add one first (e.g. `f"{source}#{chunk_index}"`). Retrieval metrics are meaningless without ids.

## Step 2 - Write the adapter

Create `evals/rag_adapter.py` exposing exactly one function. Call the app's real code - never re-implement retrieval here, or you are evaluating a copy.

```python
def answer(question: str) -> dict:
    # returns {"answer": str, "contexts": [str, ...], "context_ids": [str, ...]}
```

`contexts` and `context_ids` must be in rank order, same length.

## Step 3 - Build the golden set

Use `templates/golden_set.jsonl` as the format. One JSON object per line:

```json
{"id": "q001", "question": "...", "expected_answer": "...", "relevant_ids": ["doc.pdf#12"], "type": "lookup"}
```

Rules that keep the set honest:
- **20 questions minimum**, 50 is better. Mix types: `lookup` (single fact), `calculation`/`multi_hop` (needs 2+ chunks), `out_of_scope` (answer should refuse; `relevant_ids: []`).
- At least 15% `out_of_scope` - this is where RAG apps hallucinate most.
- Write expected answers from the source documents, not from the app's output. Copying the app's answer bakes its mistakes into the test.
- Get `relevant_ids` by searching the corpus yourself, not by trusting what the retriever returned.
- If drafting questions with an LLM, have a human verify every expected answer before committing.
- Write each question from one gold passage and store that passage's id in `relevant_ids`. A check that only asks "was any chunk of the right document (or type) retrieved?" is not hit@k, and on a real project it passed by construction for 72% of questions.
- Freeze the set before you tune against it (commit it, record its hash). Every fix you choose by reading its failures makes it less of a test set.

## Step 4 - Choose a judge model

DeepEval metrics need an LLM judge. Defaults, cheapest first:
- **Groq** (free tier, OpenAI-compatible): set `JUDGE_BASE_URL=https://api.groq.com/openai/v1`, `JUDGE_API_KEY`, `JUDGE_MODEL` (a 70B-class model).
- **Ollama** locally: `JUDGE_BASE_URL=http://localhost:11434/v1`, any key, a 7B+ instruct model.
- OpenAI: leave `JUDGE_BASE_URL` unset.

Never use the same small model as both generator and judge - it grades its own homework.
Judges smaller than ~8B frequently return malformed JSON; `run_eval.py` retries and records the failure instead of crashing the run.

## Step 5 - Run

```bash
pip install deepeval openai
python evals/run_eval.py --golden evals/golden_set.jsonl --k 5
```

The script computes, per question and averaged:

| Metric | What it tells you | Needs judge? |
|---|---|---|
| Hit@k | Did any relevant chunk make the top k? | No |
| MRR | How high was the first relevant chunk? | No |
| Faithfulness | Is every claim supported by retrieved context? Not scored for refusals | Yes |
| Answer relevancy | Does the answer address the question? | Yes |
| Citation validity | Do `[N]` markers point to chunks that exist? | No |
| Refusal rate | Did out-of-scope questions get refused? | No (keyword check) |
| False-refusal rate | Did *answerable* questions get refused? | No (keyword check) |

Citation validity is checked **before** the judge call: an answer citing `[7]` when only 5 chunks were retrieved is wrong regardless of what the judge says.

Every run also writes `evals/results.jsonl`: one line per question with the answer, the retrieved ids and text, and the scores. Both files are rewritten after every question, so a run that dies part-way (rate limit, timeout) still leaves results, marked PARTIAL. A question whose app call fails is recorded with its error, listed in `results.md`, excluded from the averages, and makes the run exit non-zero: averages over only the questions that worked would otherwise look better than the app is. Keep it. It lets you re-score a run with a fixed metric, or a different judge, without paying to regenerate the answers.

## Step 6 - Check the metrics before you trust them

A metric that can't fail looks like good news. Before reading any result, check each metric:

1. **Can it fail?** Count the questions where it passes by construction (no relevant ids, a fallback that always returns true, a check any chunk satisfies). Those rows must be excluded, as `run_eval.py` does by returning `None`, not counted as passes.
2. **Does it move when it should?** Remove a component you know matters (the reranker, half of `k`, the refusal instruction) and re-run. A retrieval metric that barely moves while accuracy collapses is measuring something else.
3. **Does it reward refusals?** Faithfulness judges score "I don't know" as fully supported, so a system that refuses more looks *more* faithful. Always report the false-refusal rate on answerable questions next to faithfulness and next to the out-of-scope refusal rate. Either one alone rewards refusing everything.
4. **Is the check stricter than the thing it checks?** Exact string matching on citations rejects correct answers that copy an id with different spacing or spelling. Normalise case, whitespace and punctuation before comparing, and count *invented* citations (an id that exists nowhere in the corpus) separately from *unverifiable* ones (real, but not retrieved for this question).
5. **What is the noise floor?** Run the identical configuration twice. Differences smaller than the rerun's flips are not results.
6. **Run negative controls.** Checks 1 to 5 find problems by reading results; controls test the metrics directly. Each control is a pair scored from fixed inputs, with no app call: the clean twin must pass and the broken twin must fail with its own signature, so a judge that accepts everything fails the broken halves and one that rejects everything fails the clean halves. Build one pair of each kind from your own corpus (format in `templates/negative_controls.jsonl`):

| Kind | Broken twin | Must show |
|---|---|---|
| `irrelevant_passage` | gold chunk replaced by a near miss (same topic and wording, different rule) | hit 0, faithfulness ≤ 0.5 |
| `missing_evidence` | decisive evidence removed, answer is a refusal | hit 0 and refused: a retrieval miss, not a generation fault |
| `unsupported_claim` | an on-topic claim the chunk never makes, e.g. an invented fee | faithfulness ≤ 0.75, relevancy still ≥ 0.7 |
| `wrong_refusal` | a refusal although the gold chunk was retrieved | false refusal flagged, relevancy ≤ 0.5 |

   Check in code that a broken twin's invented claim really is absent from its context, and keep it on topic, or it fails for the wrong reason. Run `python evals/run_eval.py --controls evals/negative_controls.jsonl`; it writes `evals/controls.md` and exits non-zero if any pair misbehaves. Run it in CI and after any judge, prompt or DeepEval upgrade, since that is when a judge quietly starts rewarding refusals again.

   On a real legal RAG app, the first run of these controls failed 2 of 4: DeepEval's faithfulness metric by default counts claims the context neither supports nor contradicts as faithful, so an invented fee and a near-miss passage both scored 1.0. `run_eval.py` now counts those claims as unfaithful, and does not score faithfulness for refusals at all (by default a refusal scores 1.0; strictly it scores 0.0, punishing correct ones), leaving refusals to the refusal and false-refusal checks.

## Step 7 - Change one thing, measure it properly

- **Screen retrieval changes without the judge.** `--retrieval-only` is deterministic and costs nothing, so try rerankers, `k` and chunking there first, and pay for a judged run only for the winner. Check once that the retrieval-only numbers match the full run's.
- **Compare paired runs, not averages.** Same questions, same judge, same settings, one change. Report the flips (better / worse per question) and an interval, not just two percentages. A +4 point average can be 17 questions better and 7 worse.
- **Never switch judges mid-comparison.** If your judge budget runs out, run `--retrieval-only` now: `results.jsonl` keeps every answer and its retrieved text, so the same judge can score them later instead of a different one scoring half the runs.
- **For deterministic components** (routers, guards, regex classifiers), diff their decisions on the whole golden set before and after a change. You should see exactly the rows you meant to change, and nothing else.
- **Read the losses.** For each question that got worse, check whether the gold chunk was still retrieved. If it was, the cause is generation or judge noise, not retrieval.

## Step 8 - Read the results like an engineer

Diagnose from the pattern, not the average:
- **Low Hit@k, high faithfulness** -> retrieval problem. Try hybrid (BM25 + vector), smaller chunks, or a reranker.
- **High Hit@k, low faithfulness** -> generation problem. Tighten the prompt ("answer only from context"), lower temperature, or the model is too small.
- **Good scores but out-of-scope questions answered** -> missing refusal instruction or no similarity threshold.
- **One question type failing** (e.g. `calculation`) -> needs a tool or structured step, not better retrieval.

Change one thing, re-run, compare `results.md`. Commit `results.md` so the history shows the improvement.

## Step 9 - CI gate (optional)

Use `templates/rag-eval.yml`. It runs the retrieval-only metrics on every PR (free, fast) and the full judge run on `main`. Thresholds are passed as flags; `run_eval.py` exits non-zero when any average falls below them.

Start thresholds at about 5 points below the current baseline, then raise them as the app improves.

## Common mistakes

- Evaluating with `k` different from production `k`.
- Golden set written by looking at app outputs (inflates every score).
- Reporting only averages - always include the per-type breakdown.
- Running the judge at temperature > 0 (scores drift between runs).
- Treating a fix found by reading test-set failures as a test-set result. Label it post-hoc, and confirm it on questions written afterwards.
