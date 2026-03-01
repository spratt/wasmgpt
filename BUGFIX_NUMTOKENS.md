# Bug: vocab size mismatch between training and inference

## Symptom

Inference cannot load the checkpoint. `loadCheckpoint` returns -2
(config mismatch). Inference generates output using random weights.

## Observed data

### The mismatch

| Source | vocabSize |
|--------|-----------|
| `vocab.tsv` (line count) | 875 |
| `vocab.tsv` (max ID + 1) | 875 |
| `vocab.tsv` (unique IDs) | 875 |
| Checkpoint header (savedVocab) | 756 |
| Training stderr ("total") | 756 |
| Inference stderr ("vocabulary") | 875 |

Training writes `vocabSize` from `getBpeNextId()` (756).
Inference computes `totalVocab` as `max ID + 1` from vocab.tsv (875).
The checkpoint validator rejects the load because 875 != 756.

### Root observation

`buildBpeVocab` is called in both `train-bpe.ts` and `train.ts` with
the same `startId` (568) and the same `vocab.size` (568). But it
produces different results:

| | train-bpe | train |
|---|---|---|
| merges.length | 256 | 135 |
| bpeVocab.size | 311 | 190 |
| bpeNextId | 875 | 756 |

`build/merges.tsv` has 256 lines. `train-bpe.ts` passes 256 merges
(generated in memory by `trainBpe`). `train.ts` passes only 135
(loaded from `merges.tsv` via `parseMerges`).

`parseMerges` is truncating the merge file, losing 121 merge rules.

### parseMerges diagnostics

| Measurement | Value |
|---|---|
| `build/merges.tsv` byte count (`wc -c`) | 2386 |
| `build/merges.tsv` line count (`wc -l`) | 256 |
| `readString()` text.length | 2384 |
| `text.split("\n")` lines.length | 136 |
| skipped (empty line) | 1 |
| skipped (no tab) | 0 |
| parsed merges | 135 |

The file has 256 newlines but `split("\n")` only finds 135 of them.
The merge file contains tokens like `\0`, `\00`, `\00\00` (literal
backslash-zero sequences representing WAT string escapes).
`readString()` or `split("\n")` is misinterpreting some `\n`
subsequences within these tokens as something other than newlines,
causing lines to be joined together and the tab delimiter to be lost.
