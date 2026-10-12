# NLQTSBench-Styles — Rewriting Specification (v1, 2026-09-20)

You are rewriting natural-language queries from NLQTSBench so that the SAME intent is expressed
in four different linguistic styles. The answer key is reused unchanged, so a rewrite is
admissible ONLY if the original answer remains the one correct answer to it.

## 0. The four styles, and why exactly these four

The canonical (template) query sits at the origin of two axes. Each style moves along one axis
in one direction:

| axis | direction | style | phenomenon |
|---|---|---|---|
| lexis | toward lay / everyday language | **lexical** | synonym substitution within everyday register; syntax unchanged |
| lexis | toward specialist language | **technical** | register shift to domain / analytics vocabulary |
| syntax | toward less structure | **keyword** | compression to a search-box fragment; content words only |
| syntax | toward more structure | **conversational** | expansion into a spoken-register request with framing and hedges |

Each style is designed to stress one stage of the pipeline: lexical → qualifier grounding;
technical → vocabulary mapping; keyword and conversational → intent parsing. Do not blend styles.

## 1. INVARIANTS — copy these verbatim into every rewrite

1. **The output-format suffix.** Every question ends with a parenthesised instruction beginning
   `(Output format:` … `)`. Copy it character-for-character, as the final sentence, in all
   four styles. Never paraphrase it, never drop it, never move it.
2. **Identifiers and values.** Channel identifiers (e.g. `67`, `MULL`, `441_down_2021`),
   timestamps, dates, years, months, numeric thresholds (`4.287`), window lengths (`59-day`),
   and `k` in "top-k" must appear exactly as in the original. You may change the words
   around them, never the tokens themselves.
3. **L4 (Insight Synthesis) structure.** The line `Please use ONLY the following phrases…`
   with its six phrases, and the numbered report structure that follows, are preserved
   verbatim. Only the first sentence (`Analyze the behavior of channel X for the period Y.`)
   is rewritten.
4. **Meaning.** The task type, the target channel, the time scope, the pattern (the ordered
   sequence of segment kinds and their qualifiers), the superlative / ranking direction, and
   the number of results requested are all preserved. Never add a constraint, never remove one,
   never change the order of segments in a pattern.

## 2. Style definitions

### 2.1 lexical — same sentence, different words
* Keep the sentence structure of the original. Replace content words with everyday synonyms
  drawn ONLY from the controlled table in §3. Replace as many replaceable words as you can
  (target: every qualifier, every segment kind, every shape noun, every superlative, and the
  main verbs).
* Do NOT change clause order, do NOT add or remove clauses.
* Example — original: *identify the top-5 days where the slow fall segment is the slowest*
  → lexical: *pick out the top-5 days where the gentle decline stretch is the most gradual*

### 2.2 technical — the same request from a domain analyst
* Use hydrology / signal-analysis / time-series-analytics vocabulary: regime, gradient,
  excursion, magnitude, stationary, monotonic, recession, drawdown, baseline, amplitude,
  window, interval, series, channel. Qualifiers become measured properties
  (rapid → high-magnitude / steep gradient; slow → low-magnitude / shallow gradient;
  steady → low-variance stationary; fluctuating → high-variance / noisy).
* Formal, nominal, precise; may reorder into "For channel X, period Y: retrieve …".
* Example: *For channel 67, calendar year 2023: retrieve the top-5 days exhibiting a
  stationary regime followed by a high-magnitude negative gradient and a subsequent
  low-magnitude negative gradient, ranked by the shallowest terminal gradient.*

### 2.3 keyword — what a user types into a search box
* Content words only. Drop articles, auxiliaries, relative clauses, politeness. Use arrows,
  commas or `>` for sequence. 5–15 tokens before the output-format suffix.
* All identifiers, values and the pattern (in order) must survive.
* Example: *channel 67 2023, steady stable > rapid fall > slow fall, top 5 slowest final fall*

### 2.4 conversational — a colleague asking out loud
* Spoken register: a greeting or lead-in, hedges ("could you", "I'm looking for", "ideally"),
  the request embedded in a longer sentence, possibly split into two sentences.
* The pattern may be described in prose ("held steady for a while, then dropped fast, then kept
  falling more slowly") but every segment kind and qualifier must still be recoverable and in
  order. 25–60 words before the output-format suffix.
* Example: *Hey, could you go through channel 67 for 2023 and pull out the days where the level
  held steady for a while, then dropped fast, and then kept falling but more gently? I'd like
  the five where that last gentle stretch was the most gradual.*

## 3. Controlled synonym table (lexical style MUST draw from these; technical may use its own §2.2 terms)

| canonical | lexical synonyms (everyday) |
|---|---|
| rise / rising | climb, go up, increase, move up |
| fall / falling | drop, decline, decrease, go down |
| stable | flat stretch, level period, plateau, holding steady |
| rapid | quick, fast, sharp, steep, abrupt |
| slow / gradual | gentle, gradual, slow-moving, unhurried |
| steady | calm, smooth, even, flat |
| fluctuating | choppy, bumpy, unsteady, jittery, wobbly |
| plateau (shape) | flat top, level stretch |
| low plateau / bottom out | flat bottom, floor, trough plateau |
| upward spike | sharp jump, sudden peak, spike up |
| deep valley | deep dip, sharp trough |
| step ascent / step descent | step up / step down, jump up / jump down |
| surge | jump, spike, sudden rise |
| drought / dry-out | dry spell, dry period |
| longest | most prolonged, most extended, longest-lasting |
| highest / largest / deepest | biggest, greatest, tallest / deepest |
| slowest / fastest | most gradual / quickest |
| most fluctuating | choppiest, bumpiest, most wobbly, most unsteady, most jittery (keep the copula: "… stretch is the choppiest") |
| most similar | closest match, best match, most alike |
| identify / find | pick out, locate, spot, look for |
| dominant cycle period | main repeating cycle, strongest periodicity |
| median / average / maximum / minimum / range | middle value / mean / top value / lowest value / spread |
| first rise above / fall below | first climb past / first drop under |
| remained above / below | stayed above / below |

## 4. What NOT to do
* Do not introduce a qualifier that is not in the original (e.g. adding "rapid" to a plain "fall").
* Do not drop a qualifier that is in the original.
* Do not change numbers, units, ids, dates or the output-format suffix.
* Do not explain, comment, or add anything outside the rewritten question.
* Do not produce a rewrite that could be answered correctly by a different answer than the
  original's.

## 5. Output
For each input task, output one JSON object:
```json
{"id": "<task id>",
 "lexical": "<rewrite>", "technical": "<rewrite>", "keyword": "<rewrite>", "conversational": "<rewrite>",
 "self_check": {"ids_and_values_preserved": true, "pattern_order_preserved": true, "suffix_verbatim": true}}
```
Before emitting, re-read each rewrite against §1 and flip the corresponding self_check flag to
false if you are not certain.
