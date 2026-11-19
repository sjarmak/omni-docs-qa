# Judging documentation

The harness answers one question about a docs set: can a reader's question be
answered from it, and if not, what is missing or contradictory? This page
covers what to do with that answer, and how to judge the part the harness does
not score: the writing itself.

## 1. Reading the gap queue

A gap verdict is a lead, not a finding. Before a writer acts on one:

- **Recheck it against the current docs.** The example sets were labelled
  against the 2026-09-15 snapshot. Some gaps have since been closed. For
  example, audit log retention ("at least one year") is now stated on the
  audit logs overview.
- **Separate "undocumented" from "unsupported".** Many `not_covered`
  questions ask for features that may not exist, such as a Terraform provider
  or a Microsoft Teams delivery destination. The fix is often one sentence
  saying it is not supported. Product has to confirm that first.
- **Rule out retrieval before blaming the page.** If the answer is in the
  corpus but the system returned `not_covered`, the problem is usually
  findability: the fact is buried, split across pages, or worded differently
  from how readers ask. That is still a docs signal, but the fix is structure
  or wording, not new content. On the v2 set, BM25 at k=8 missed the
  supporting section for 12 evidence-bearing questions (listed in the README).
  Start there.
- **Treat citations spread across neighbouring pages as a duplication
  signal.** Several regression misses were correct answers that cited the
  other page holding the same fact, such as iframe restrictions on two embed
  pages. Duplicated facts drift apart. Either keep one canonical page and link
  to it, or list both in `expected_cited_page_urls`.
- **Group the gaps by pattern.** On the Omni set, about 20 gap questions ask
  for a limit the docs never state: counts, sizes and retention windows. One
  limits reference page closes more gaps than 20 separate edits.
- **Planted contradictions are a test of the system, not of the docs.**
  Never report `conflicting` results on injected sections as documentation
  problems.

## 2. Judging writing style

### Start from a published guide, not a new one

Google's technical writing course advises against writing your own house
editorial guide. It recommends adopting the
[Google developer documentation style guide](https://developers.google.com/style)
or the [Microsoft Writing Style Guide](https://learn.microsoft.com/style-guide/welcome/),
plus a short word list for product terms
([source](https://developers.google.com/tech-writing/resources)). The Google
guide sets the order of precedence: project style, then the guide, then
general references.

### Lint what is mechanical

Terminology, product names, heading case, voice and banned phrases can all be
checked by a prose linter instead of a reviewer.
[Vale](https://vale.sh/) runs offline, understands Markdown and MDX, skips
code blocks, and ships ready-made Google, Microsoft and Red Hat styles. Put
house rules in YAML next to the docs and run it in CI. This catches the
cheapest problems at no cost, and leaves human or model review for things a
rule cannot express.

### Judge a page against the job it is meant to do

[Diátaxis](https://diataxis.fr/) separates tutorials, how-to guides,
reference and explanation, because each serves a different need. Much
"unclear" writing is a page doing two of those jobs at once, such as a how-to
that stops to explain concepts. Label each page's intended form first, then
judge it against that form.

A rubric that works across forms, drawn from the sources here:

| Dimension | Question for the reviewer |
| --- | --- |
| Purpose | Does the opening say what this is and who it is for? |
| Task fit | Can a reader finish the task or find the fact without reading other pages? |
| Conciseness | Could anything be removed without losing information the reader needs? |
| Accuracy | Is every claim supported by the product? Are there no invented options, limits or defaults? |
| Structure | Do headings match how readers ask? Is each fact on one canonical page? |
| Consistency | Does it follow the style guide and word list? (Mostly the linter's job.) |
| Translation and scanning | Short sentences, consistent terms, examples that don't depend on culture? |

Keep the scale small (pass / needs work / fail, or a 4-point scale). Record
a one-line reason for every score below the top.

### Conciseness needs a reference point

"Too long" is only measurable against what was needed.
[YapBench](https://arxiv.org/abs/2601.00624) pairs each prompt with a
*minimal sufficient answer*: the shortest response raters still judge
correct, clear and sufficient. It then measures excess length beyond that.
The same idea works for docs. A writer marks the minimal sufficient version
of a section or answer, and drafts are scored by how far they exceed it
without adding needed information.

Two findings explain why this matters more now:

- **Model output has been getting longer.** Epoch AI measured responses to
  benchmark questions growing about 2.2× per year for non-reasoning models
  and 5× per year for reasoning models
  ([Epoch AI, 2025](https://epoch.ai/data-insights/output-length)).
- **Extra length often signals uncertainty.** The "verbosity compensation"
  study found that models pad answers by repeating the question, hedging and
  over-enumerating. Those padded answers were less accurate and more
  uncertain, and the gap did not shrink as models got more capable
  ([Zhang et al., 2025](https://aclanthology.org/2025.uncertainlp-main.14/)).
  When an AI-drafted section runs long, fact-check it first.

Each model family and version has its own writing habits.
Wikipedia's [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
catalogues them: stock phrases, closing summaries, "not just X, it's Y"
constructions, and over-bolded lists. It notes that what is typical of one
model is not typical of another. It is descriptive, not a rule set, but it
works as a checklist when editing AI drafts.

### Comparing models as writers

Some teams find that an older model writes tighter technical prose than
current ones. That can be measured rather than argued:

1. Fix the inputs. Take a few real pages, remove them, and keep everything a
   writer would have: the product, related pages and the style guide.
   [readme-bench](https://github.com/jarmak-personal/readme-bench) shows the
   method for READMEs: same snapshot, same one-line prompt, pinned harness
   version, and every output published unedited.
2. Have each model rewrite each page, with the same prompt and the same
   effort setting.
3. Review the outputs side by side, section by section, using the rubric
   above. Record excess length against the writer's minimal sufficient
   version.
4. Report per model: rubric results, excess length, and invented claims, the
   things the page states that the product does not do.

Use pages that are not yet published, or deliberately altered ones. Public
docs are in training data, so regenerating them measures recall as much as
writing.

## 3. If you use a model as the judge

LLM judges agree with human reviewers well enough to be useful. They have
known biases that a docs rubric will trip over unless you design around
them.

- **Length and formatting bias.** AlpacaEval found automatic judges prefer
  longer answers and answers with lists. Asking the model under test to "be
  as concise as possible" dropped its win rate from 50% to 23%, and asking for
  "as much detail as possible" raised it to 64%. A length-controlled
  correction fixed most of this
  ([Dubois et al., 2024](https://arxiv.org/abs/2404.04475);
  [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)). The bias varies by
  judge. A 2026 study found Gemini 2.5 judges favour padded answers, while
  Claude Sonnet 4 slightly favoured the concise one, and most judges preferred
  Markdown formatting
  ([Judging the Judges, 2026](https://arxiv.org/abs/2604.23178)). For a
  rubric that rewards conciseness, a length-biased judge works against you.
  Normalize formatting before judging, and include padded and truncated
  control pairs.
- **Position bias.** Judges favour whichever answer appears first or second.
  Judge every pair in both orders
  ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)). A 2026 audit of
  21 judges found high test–retest reliability alongside strong position bias
  in two production judges. Judge rankings also shifted by up to 14 places
  across benchmarks
  ([Reliability without Validity, 2026](https://arxiv.org/abs/2606.19544)).
  Consistency is not the same as being right.
- **Self-preference.** Judges can favour their own family's output. Part of
  the measured effect is the judge being wrong where the model is wrong,
  rather than recognising itself
  ([2026](https://arxiv.org/abs/2601.22548)). Either way, use a judge from a
  different family than the writer.
- **Overlap metrics don't measure quality.** BLEU, ROUGE and METEOR did not
  track expert ratings of generated documentation
  ([2024](https://arxiv.org/abs/2408.14007)). A technical writer's error
  analysis matched an LLM correctness judgment better than it matched
  deterministic metrics
  ([de Souza et al., 2025](https://aclanthology.org/2025.nodalida-1.67/)).
- **Rubrics derived from standards help.** Judging against explicit criteria
  taken from documentation standards reached Spearman ρ = 0.72 with human
  ratings, and results held across judge families
  ([CodeDoc, 2026](https://arxiv.org/abs/2610.27634)).

A working order, cheapest first:

1. Vale against the house style.
2. Human rubric review on a sample, by the docs team, with reasons recorded.
3. An LLM judge only after it matches those human labels on a held-out
   sample. Use pairwise comparison in both orders, a judge from a different
   family than the writer, and length-controlled or length-matched pairs.
   Report agreement as Cohen's κ, not raw percent.
4. Rerun the human sample whenever the judge model or prompt changes. Judge
   behaviour shifts between versions.
