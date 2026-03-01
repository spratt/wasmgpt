# Multi-file corpus plan

## Problem

The training pipeline currently accepts a single annotated WAT file as
corpus input. This is insufficient for two reasons:

1. **wasmgpt has 4 entry points**, each compiling to a separate `.wasm`
   with its own source map. Only `index.ts` is annotated today, covering
   4 of 12 source files. The other 8 files (model.ts, tensor.ts,
   train.ts, infer.ts, checkpoint.ts, config.ts, io.ts, train-bpe.ts)
   are absent from the corpus.

2. **Other projects** (starting with watnot) should contribute to the
   corpus. watnot has 5 source files and already builds to WASM with a
   source map.

## Source map coverage

Each entry point's source map includes only the files it transitively
imports:

| Entry point   | Source files in map |
|---------------|---------------------|
| `index.ts`    | index, lexer, vocabulary, bpe |
| `train-bpe.ts`| train-bpe, lexer, vocabulary, bpe, io |
| `train.ts`    | train, lexer, vocabulary, bpe, io, config, model, tensor, checkpoint |
| `infer.ts`    | infer, lexer, vocabulary, bpe, io, config, model, tensor, checkpoint |

The union of all 4 covers all 12 source files. Shared files (lexer,
bpe, vocabulary) appear in multiple maps — their comments will be
injected independently into each annotated WAT, which is fine (the
model sees the same functions in different call-graph contexts).

watnot has a single entry point covering all 5 of its source files.

## Changes

### Step 1: Update watnot's annotation to include all source files

watnot's `watnot-annotated.wat` currently has no comments. Its
package.json needs `wat` and `annotate` scripts, and the annotate step
must list all 5 source files:

```
"wat": "wasm2wat build/watnot.wasm --fold-exprs --output build/watnot.wat --offset-map build/watnot.offsets.json"
"annotate": "wasmtime --dir . build/watnot.wasm build/watnot.wasm.map build/watnot.wat build/watnot.offsets.json src/index.ts src/comments.ts src/injector.ts src/offsetmap.ts src/sourcemap.ts > build/watnot-annotated.wat"
```

### Step 2: Build and annotate all wasmgpt entry points

Add per-entry-point scripts to wasmgpt's package.json. Each entry point
needs: compile to .wasm (with --sourceMap), disassemble to .wat (with
--offset-map), annotate (listing source files from its source map).

New build outputs:
- `build/wasmgpt.annotated.wat` (from index.ts)
- `build/train-bpe.annotated.wat` (from train-bpe.ts)
- `build/train.annotated.wat` (from train.ts)
- `build/infer.annotated.wat` (from infer.ts)

The annotate script for each entry point passes only the source files
present in that entry point's source map.

### Step 3: Update `train-bpe.ts` to accept multiple corpus files

Currently:
```
wasmtime --dir . build/train-bpe.wasm build/wasmgpt.annotated.wat
```

Change to accept multiple file paths. Tokenize each file and combine
the unknown token lists before BPE training:

```
wasmtime --dir . build/train-bpe.wasm build/wasmgpt.annotated.wat build/train-bpe.annotated.wat build/train.annotated.wat build/infer.annotated.wat watnot/build/watnot-annotated.wat
```

Implementation: loop over `args[1..]`, tokenize each, accumulate into
the combined unknown tokens list, then run `trainBpe` once on the
combined set.

### Step 4: Update `train.ts` to accept multiple corpus files

Currently:
```
wasmtime --dir . build/train.wasm build/wasmgpt.annotated.wat build/merges.sexp
```

Change to accept merges path as the last argument (or a flag), and
treat all other args as corpus files. Tokenize and encode each file,
concatenate all token IDs (with BOS between each file) into a single
training sequence.

```
wasmtime --dir . build/train.wasm build/merges.sexp build/wasmgpt.annotated.wat build/train-bpe.annotated.wat build/train.annotated.wat build/infer.annotated.wat watnot/build/watnot-annotated.wat
```

Implementation: identify merges path (first .sexp arg), treat remaining
args as corpus files. For each file: tokenize, encode, wrap with BOS,
append to allIds.

### Step 5: Update package.json scripts

```json
"wat:index": "wasm2wat build/wasmgpt.wasm --fold-exprs --output build/wasmgpt.wat --offset-map build/wasmgpt.offsets.json",
"wat:train-bpe": "wasm2wat build/train-bpe.wasm --fold-exprs --output build/train-bpe.wat --offset-map build/train-bpe.offsets.json",
"wat:train": "wasm2wat build/train.wasm --fold-exprs --output build/train.wat --offset-map build/train.offsets.json",
"wat:infer": "wasm2wat build/infer.wasm --fold-exprs --output build/infer.wat --offset-map build/infer.offsets.json",

"annotate:index": "wasmtime --dir . watnot/build/watnot.wasm build/wasmgpt.wasm.map build/wasmgpt.wat build/wasmgpt.offsets.json src/index.ts src/lexer.ts src/vocabulary.ts src/bpe.ts > build/wasmgpt.annotated.wat",
"annotate:train-bpe": "wasmtime --dir . watnot/build/watnot.wasm build/train-bpe.wasm.map build/train-bpe.wat build/train-bpe.offsets.json src/train-bpe.ts src/lexer.ts src/vocabulary.ts src/bpe.ts src/io.ts > build/train-bpe.annotated.wat",
"annotate:train": "wasmtime --dir . watnot/build/watnot.wasm build/train.wasm.map build/train.wat build/train.offsets.json src/train.ts src/lexer.ts src/vocabulary.ts src/bpe.ts src/io.ts src/config.ts src/model.ts src/tensor.ts src/checkpoint.ts > build/train.annotated.wat",
"annotate:infer": "wasmtime --dir . watnot/build/watnot.wasm build/infer.wasm.map build/infer.wat build/infer.offsets.json src/infer.ts src/lexer.ts src/vocabulary.ts src/bpe.ts src/io.ts src/config.ts src/model.ts src/tensor.ts src/checkpoint.ts > build/infer.annotated.wat",

"corpus": "npm run build && npm run build:train-bpe && npm run build:train && npm run build:infer && npm run wat:index && npm run wat:train-bpe && npm run wat:train && npm run wat:infer && npm run annotate:index && npm run annotate:train-bpe && npm run annotate:train && npm run annotate:infer",
"corpus:watnot": "cd watnot && npm run build && npm run wat && npm run annotate && cd ..",

"train:bpe": "npm run corpus && npm run corpus:watnot && wasmtime --dir . build/train-bpe.wasm build/wasmgpt.annotated.wat build/train-bpe.annotated.wat build/train.annotated.wat build/infer.annotated.wat watnot/build/watnot-annotated.wat",
"train": "npm run build:train && wasmtime --dir . build/train.wasm build/merges.sexp build/wasmgpt.annotated.wat build/train-bpe.annotated.wat build/train.annotated.wat build/infer.annotated.wat watnot/build/watnot-annotated.wat"
```

### Step 6: Update clean script

Add the new WAT/offsets/annotated files to the clean target.

## Corpus size estimate

| Source | Lines (current) |
|--------|----------------|
| wasmgpt index | 11,803 |
| wasmgpt train-bpe | ~similar (shared code) |
| wasmgpt train | ~larger (model, tensor) |
| wasmgpt infer | ~larger (model, tensor) |
| watnot | 24,568 |

Shared source files (lexer, bpe, vocabulary) will appear multiple times
across entry points, but in different compilation contexts. Total corpus
will be significantly larger than the current 11,803 lines.

## Verification

1. `npm test` — all existing tests pass (no test changes needed)
2. `npm run corpus` — produces 4 annotated WAT files for wasmgpt
3. `npm run corpus:watnot` — produces annotated WAT for watnot with comments
4. `npm run train:bpe` — trains BPE on combined corpus
5. `npm run train` — trains model on combined corpus
6. Spot-check: `grep ';;' build/train.annotated.wat | head` shows injected comments from model.ts/tensor.ts
7. Spot-check: `grep ';;' watnot/build/watnot-annotated.wat | head` shows injected comments from watnot sources
