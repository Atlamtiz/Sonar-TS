# NLQTSBench-Styles — Judge Specification (layer 2 of validation)

You are judging whether a rewritten query is an admissible member of NLQTSBench-Styles. The
deterministic checker (layer 1) has already verified that identifiers, numbers, timestamps, the
output-format suffix, and the ordered pattern slots survive. Your job is what a regex cannot do.

Read `STYLE_SPEC.md` first. Then, for each rewrite, answer three questions.

## Q1 — Same question? (intent preservation)  → `intent: pass|fail`
The original's answer key must be the one correct answer to the rewrite. Fail if the rewrite:
- asks for a different quantity, a different ranking direction, a different number of results;
- adds a constraint the original lacks (a qualifier, a threshold, a time restriction) or drops one;
- changes the ORDER of segments in a composite pattern, or lets a qualifier drift to a different
  segment (e.g. original: "slow fall, then fall" — a rewrite where the second fall becomes the
  slow one fails);
- introduces ambiguity that admits a second correct answer;
- for L4, alters anything after the first sentence.
Be strict on qualifiers: "rapid" ≠ "sudden onset" if the original meant rate; "steady stable"
means a low-variance flat segment, not merely "stable".

## Q2 — Right style? (style adherence)  → `style: pass|fail`
- lexical: sentence structure of the original preserved; content words replaced by everyday
  synonyms; NOT a paraphrase, NOT more formal, NOT more casual.
- technical: specialist / analyst register; nominal, precise; qualifiers rendered as measured
  properties. Fail if it is just the original with one jargon word swapped in.
- keyword: a fragment — no full clauses, no politeness, 5–15 tokens before the suffix.
- conversational: spoken register with framing; 25–60 words before the suffix; the request
  embedded in discourse. Fail if it is a single formal sentence with "Hi" bolted on.
A rewrite that is a good query but in the wrong style FAILS Q2.

## Q3 — Fluent and natural? (quality)  → `quality: pass|fail`
Fail only for genuinely broken output: ungrammatical in a way a user would not produce, garbled,
truncated, or containing artefacts (JSON fragments, meta-commentary, the word "rewrite").

## Verdict
`accept` iff all three pass. Otherwise `reject` with a one-line `reason` naming the failing
question and the specific problem, phrased so a rewriting agent can fix it in one attempt
(e.g. "Q1: qualifier 'slow' moved from segment 1 to segment 2", "Q2: keyword style has full
clause 'identify the days where'").

Do not reject for taste. Do not reject a lexical rewrite for using a synonym that is in
STYLE_SPEC §3. Do not accept a rewrite because it is close — the dataset must be exactly right.
