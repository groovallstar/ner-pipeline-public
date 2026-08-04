---
name: explain-diff
description: Use when the user asks for a rich explanation of a code change, diff, branch, or PR. Produces a plain Markdown file (mermaid diagrams, self-check quiz) that renders natively in GitHub and VS Code.
metadata:
  source: https://gist.github.com/geoffreylitt/a29df1b5f9865506e8952488eac3d524
  adaptedFor: ner_pipeline
  variant: markdown-first — HTML 원본을 이 저장소 컨벤션(mermaid·in-repo 렌더)에 맞게 재작성
---

# Explain Diff

Please make me a rich explanation of the specified code change, as a Markdown document.

It should have these sections:

- Background: Explain the existing system relevant to this change. (You should broadly explore surrounding code for this.) We don't know how much the reader already knows, so include a deep background for beginners (note that it can be skipped if the reader is already familiar), and then a more narrow background directly relevant to the change.
- Intuition: Explain the core intuition for the code change. The focus here is to explain the essence, not the full details. Use concrete examples with toy data. Use figures and diagrams liberally.
- Code: Do a high-level walkthrough of the changes to the code. Group/order the changes in an understandable way.
- Quiz: Come up with five questions that test the reader's knowledge of this PR. Medium difficulty — hard enough that you must understand the substance of the PR to answer, but not gotchas. The goal is to help the reader confirm they've actually understood. Write each as a multiple-choice question with lettered options; do not reveal the answer beside the question — collect the answers and explanations in a single answer-key section at the end so the reader can attempt them first.

Format:

- Output a single self-contained Markdown (`.md`) file: a top `#` title, a short `>` blockquote header (commit/target), a Markdown table-of-contents linking each `##` section, then the sections. Put the file outside the code repo with a date-prefixed filename so it stays time-sorted and out of version control — e.g. `/tmp/YYYY-MM-DD-explanation-<slug>.md`. It previews natively in VS Code from anywhere; move it into `docs/reports/` only if the reader decides to keep it.
- Please write with the clarity and flow of Martin Kleppmann, making it engaging and written in classic style. Transitions between sections should be smooth.
- Write ALL prose in Korean — section headers, narrative, callouts, quiz questions, and answer key. Keep code, identifiers, and established domain jargon (BIO, span, pooled, fold, σ, fingerprint, RULER, F1, seqeval 등) in their original form, and gloss each jargon term in one short line on first use. Avoid stiff translationese; use natural Korean sentence structure. (Code comments/logs stay in the language the surrounding codebase uses.)
- Diagrams — draw them with **mermaid** (```mermaid ` `flowchart`), per this project's convention. Pick a small number of diagram families and reuse them; a data-flow `flowchart` between components is the workhorse — **put example data in the node labels** (e.g. `data_fingerprint=a1b2c3…`). In flowchart boxes, describe what each step *does* in plain language rather than naming functions/files — list the function and file names in a table or prose beneath the figure. Wrap box text in double quotes so parentheses don't break parsing. Don't force a diagram where a table or list is clearer (decision tables, folder trees, simple side-by-side comparisons) — use a Markdown table there.
- Code walkthrough — use fenced code blocks (```python …).
- Callouts for key concepts, definitions, and important edge cases — use a blockquote led by a bold label: `> **핵심 …**`, `> **엣지케이스 …**`, `> **참고 …**`. Plain text only; no decorative symbols or emoji.
- Quiz — write each question with lettered options (A–D), varying which option is correct across the five questions so position isn't a tell. Do not print the answer next to the question. Collect all answers in one `## 정답 및 해설` section at the end: give the correct letter and a one-line reason it's right (and, where useful, why a tempting wrong option fails). Plain Markdown only. Pattern:

  ```markdown
  **1. 질문 문장…?**

  - A. 보기 텍스트
  - B. 보기 텍스트
  - C. 보기 텍스트
  - D. 보기 텍스트
  ```

  ...and at the end of the document:

  ```markdown
  ## 정답 및 해설

  1. **C** — 왜 맞는지 한 줄. (흔한 오답 A 는 … 때문에 틀림.)
  2. **A** — …
  ```
