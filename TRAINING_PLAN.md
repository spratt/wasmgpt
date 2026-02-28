# Plan: Training Pipeline

## Context

The tokenizer (lexer + vocabulary + BPE) converts WAT source text into a flat array of integer token IDs. The training pipeline takes that array and trains a GPT-2 model on it. Inference is deferred — it consumes a trained checkpoint but doesn't need to exist for training to work.

The pipeline follows consgpt.lisp's structure, ported to AssemblyScript.

## Step 1: `src/autograd.ts` — Scalar autograd

Port consgpt.lisp's `autograd.lisp`. Domain-independent.

- **`Val` class** with fields: `data: f32`, `grad: f32`, `children: Array<Val>`, `localGrads: Array<f32>`
- **Differentiable operations**: `vadd`, `vmul`, `vpow`, `vlog`, `vexp`, `vrelu`, `vneg`, `vdiv`, `vsum`
- **`backward(loss: Val): void`** — DFS topological sort, reverse walk accumulating gradients, then detach graph (nil out children/localGrads for GC)

Key change from consgpt.lisp: `f32` precision instead of `double-float`.

## Step 2: `tests/autograd.test.ts` — Autograd tests

- Forward pass correctness for each operation
- Backward pass: gradient of `a + b`, `a * b`, `a^n`, `log(a)`, `exp(a)`, `relu(a)`
- Chain rule through composed expressions
- Graph detachment after backward (children are null)

## Step 3: `src/model.ts` — GPT-2 model

Port consgpt.lisp's `model.lisp`.

- **Hyperparameters**: `nEmbd=64`, `nLayer=2`, `nHead=4`, `headDim=16`, `blockSize=256`
- **PRNG**: LCG for deterministic weight initialization
- **Weight initialization**: `makeMatrix(nout, nin)` — Gaussian scaled by 0.02
- **State dictionary**: `Map<string, Array<Array<Val>>>` keyed by layer names
- **`initModel(vocabSize: i32): void`** — allocate all weight matrices, weight tying (no separate `lm_head`)
- **Architecture functions**: `linear`, `softmax`, `rmsnorm`
- **`gpt(tokenId, posId, cacheKeys, cacheVals): Array<Val>`** — forward pass returning logits

## Step 4: `src/train.ts` — Training entry point

Port consgpt.lisp's `train.lisp`. This is the CLI script that ties everything together.

- Call `initVocabulary()` to populate Pass 1 vocab, then `buildBpeVocab(merges, nextId)` with trained or loaded merges
- Read training data from stdin, tokenize with `tokenize()` → look up each token in `vocab` (Pass 1) or `bpeEncodeToken()` (Pass 2) → flat ID array
- Initialize model
- Training loop:
  - Batch extraction (contiguous window from corpus)
  - Forward pass through `gpt()`
  - Cross-entropy loss via `softmax` + `vneg(vlog(...))`
  - `backward(loss)`
  - Adam update with LR decay and bias correction
  - Print loss per step
- Checkpointing every N steps

Checkpointing format TBD — could write JSON or a binary format since we don't have S-expressions. JSON is simpler for debugging.

## Step 5: Integration test

A smoke test that:
1. Tokenizes a small WAT snippet
2. Initializes a model with the vocabulary size
3. Runs one forward pass
4. Computes loss and runs backward
5. Verifies gradients are non-zero

This validates the full pipeline without needing a full training run.

## Verification

1. `npm test` — all unit tests pass (autograd, model, integration)
2. `npm run build` — compiles
3. Training smoke test: tokenize a small WAT file, run a few training steps, verify loss decreases

## Module dependency order

```
lexer.ts (done)
  └─ vocabulary.ts (done)
  └─ bpe.ts (done)
autograd.ts (independent)
model.ts (depends on autograd)
train.ts (depends on all above)
```

## AssemblyScript constraints to watch for

- No closures — all functions top-level or class methods
- No `for..of` — use index-based loops
- Field initializers required on classes
- Sort comparators must be top-level functions
- `<` after `.get()` is interpreted as generic type — store `.get()` result in local variable first
- `f32` arithmetic: may need explicit `f32()` casts to avoid implicit promotion
- `Map` keys must be value types or strings — no object keys
