# WasmGPT: A GPT Trained on WebAssembly

## Overview

This project builds a GPT-style language model whose domain is WebAssembly (WASM/WAT). The model is trained to perform **code completion** — given a WAT prefix, generate a valid and plausible continuation. The project is implemented in **AssemblyScript** (or another language that compiles to clean, readable WAT), so that the project's own source code can be included in the training corpus.

The long-term vision includes a natural language interface, achieved by using a general-purpose LLM to annotate existing WAT with human-readable descriptions, then fine-tuning on those pairs. This is deferred to a future phase.

---

## Goals

- Train a GPT-style model from scratch on WebAssembly source
- Implement the model itself in a language that compiles to readable WAT (AssemblyScript preferred)
- Run inference in the browser using WASM + WebGPU
- Generate the training corpus from real, annotated WAT rather than raw binary or synthetic data

---

## Template: consgpt.lisp

This project is based on [consgpt.lisp](https://github.com/spratt/consgpt.lisp), a GPT trained on Common Lisp source code. WAT and Common Lisp are both S-expression languages — the structural similarity means the tokenizer architecture, model design, and training pipeline can be directly reused with targeted changes.

### What carries over from consgpt.lisp

The following components port with only surface-level changes (data types, syntax):

- **Hybrid tokenizer architecture** — consgpt.lisp uses a two-pass tokenizer: Pass 1 assigns fixed integer IDs to all 978 ANSI CL symbols plus syntax tokens, Pass 2 uses BPE for user-defined names, numbers, and strings. WasmGPT replaces the 978 CL symbols with 542 WASM instruction mnemonics (plus syntax tokens and structural keywords, ~553 unique tokens total) but keeps the same two-pass design. Both projects face the same split: a closed, enumerable set of language primitives plus open-ended user content.

- **BPE training and encoding** — the BPE algorithm (character splitting, pair counting, iterative merging, subword encoding) is language-agnostic. The merge training (`train-bpe.lisp`) and encoding (`bpe.lisp`) logic ports directly.

- **GPT-2 model architecture** — consgpt.lisp implements a GPT-2 style transformer with RMSNorm, weight tying (`lm_head` shares weights with `wte`), multi-head self-attention with KV cache, and a ReLU MLP block. The architecture is identical for WasmGPT; only the vocabulary size changes.

- **Autograd** — consgpt.lisp uses scalar autograd (`Val` per f32 value) with DFS topological sort backward pass. WasmGPT replaces this with tensor-based autograd (one `Tensor` per matrix/vector), but the backward algorithm is identical: topological sort, reverse walk, gradient accumulation, graph detachment.

- **Adam optimizer** — with linear LR decay, bias correction, and per-parameter first/second moment tracking.

- **Training loop** — batch extraction, forward pass, loss computation (cross-entropy via negative log probability), backward pass, Adam update, checkpointing.

- **Inference loop** — autoregressive generation with temperature-scaled sampling, KV cache with graph detachment for memory management.

### What changes

| Component | consgpt.lisp | WasmGPT |
|---|---|---|
| Implementation language | Common Lisp (SBCL) | AssemblyScript (compiled to WASM) |
| Fixed vocabulary | 978 ANSI CL symbols + 10 syntax tokens + 15 LOOP keywords + 13 common keywords = 1,016 | 542 WASM instruction mnemonics + 2 syntax tokens + 17 structural keywords + 7 type names = ~553 unique tokens (568 registered, 15 keyword/instruction overlaps deduplicated) |
| Lexer | Common Lisp reader (atoms, strings, dispatch macros, `#\|...\|#` block comments) | WAT S-expression parser (mnemonics, immediates, identifiers, `;;` comments) |
| Numeric precision | `double-float` (64-bit) | `f32` (32-bit) — matches WASM native type and WebGPU shader constraints |
| Data structures | Hash tables, cons lists, defstruct | Maps, Arrays, classes with field initializers |
| Runtime | SBCL native compilation, 16GB heap | WASM + WebGPU in the browser |
| Autograd | Scalar (`Val` per f32, millions of graph nodes) | Tensor (one `Tensor` per matrix, dozens of graph nodes) |
| Checkpointing | S-expressions written with `*print-readably*` | Binary (raw `fd_write`/`fd_read` via WASI, `build/model.bin`) |
| Memory management | SBCL GC with graph detachment after backward | AS GC with same detachment pattern |

### Lexer adaptation

The consgpt.lisp lexer (`lexer.lisp`) handles CL-specific syntax: `#'` (function), `#\` (character), `#(` (vector), `` ` `` (backquote), `,` and `,@` (unquote). The WAT lexer is simpler — WAT has only parentheses, atoms (instruction mnemonics, `$identifiers`, numeric literals), and `;;` comments. No dispatch macros, no reader macros, no string escapes beyond what WAT defines.

The WAT lexer is implemented in `src/lexer.ts` and exports a single function:

```typescript
export function tokenize(text: string): Array<string>
```

It splits WAT text into a flat list of token strings. Comments are stripped. Tokens are returned verbatim (case-sensitive — unlike consgpt.lisp's lexer which upcases atoms, since WAT is case-sensitive). Both Pass 1 and Pass 2 of the hybrid tokenizer consume the lexer's output.

**Token types:**
- `(` and `)` — single-character tokens
- `;;` line comments — stripped
- `(; ... ;)` block comments — stripped, with nesting support via depth counter
- `"..."` string literals — returned including quotes, backslash escapes handled
- Atoms — everything else (mnemonics, identifiers, numbers, keywords) until a terminator: whitespace, `(`, `)`, `"`, or `;`

**Dispatch order:** line comment → block comment → `(` → `)` → `"` → atom. The `(;` check precedes the `(` check so block comment starts are not emitted as open-paren tokens. Including `;` in the terminator set ensures atoms stop before `;;` comments even without intervening whitespace.

All helper functions are top-level (no closures, per AssemblyScript constraints). Character tests use `charCodeAt()` with numeric constants.

### Model size

Total parameters = VE + BE + L(12E² + 4E), where V = vocab size, E = embedding dimension, B = context length, L = layers. This formula reflects weight tying: the token embedding (`wte`, V×E) is reused as the output projection (no separate `lm_head`), saving VE parameters. Both operations relate tokens to the same embedding space, so sharing enforces symmetry and reduces parameter count.

wasmgpt's vocabulary (876 tokens after BPE) is smaller than consgpt.lisp's (1,382), reducing the embedding matrix.

**Configurations considered (V=876):**

| Config | E | L | Heads | B | Params | Checkpoint (binary, 3× for Adam) |
|---|---|---|---|---|---|---|
| **Current (Tiny)** | 64 | 2 | 4 | 256 | **~170K** | ~2 MB |
| Small | 128 | 4 | 4 | 256 | **~930K** | ~11 MB |
| Medium | 256 | 6 | 8 | 512 | **~5.0M** | ~60 MB |
| Large | 512 | 8 | 8 | 512 | **~25M** | ~300 MB |
| GPT-2 Small | 768 | 12 | 12 | 1024 | **~124M** | ~1.5 GB |

For comparison, consgpt.lisp's Tiny config (V=1,382) has 203K params. wasmgpt's smaller vocabulary saves ~33K parameters at the same architecture.

**Current config:** Tiny (E=64, L=2, H=4, B=256). Each 100-step training run completes in ~49 seconds on wasmtime, giving fast iteration before scaling up.

### Performance comparison

WasmGPT uses tensor-based autograd instead of consgpt.lisp's scalar autograd. With scalar autograd, each `f32` value is a heap-allocated `Val` node, and a single matrix multiply creates thousands of graph nodes. With tensor autograd, each weight matrix is one `Tensor` backed by a contiguous `StaticArray<f32>`, and `matmul` creates one graph node. The forward pass graph has dozens of nodes instead of millions.

**Training: 100 steps on CPU**

| Implementation | Autograd | Model | Time |
|---|---|---|---|
| wasmgpt (wasmtime) | Tensor | ~170K params, V=876 | **49s** |
| consgpt.lisp (SBCL) | Scalar | 203K params, V=1,382 | ~600s |

Even accounting for the smaller model (~0.6x the parameters), wasmgpt is roughly 7-8x faster. The speedup comes from eliminating graph overhead — millions of per-scalar heap allocations, children arrays, and local-gradient lists are replaced by a handful of tensor nodes with contiguous memory.

For reference, the microgpt benchmarks (a minimal ~4.2K parameter model used during language selection) were:

| Language | microgpt time |
|---|---|
| Common Lisp (SBCL) | 12s |
| Python | 90s |
| Clojure | 129s |
| R7RS Scheme (CHICKEN) | 170s |

wasmgpt trains a 29x larger model in only 4x the time of SBCL's microgpt.

---

## Architecture

The model follows the GPT-2 architecture as implemented in [consgpt.lisp](https://github.com/spratt/consgpt.lisp):

- Transformer with multi-head self-attention and KV cache
- RMSNorm (not LayerNorm) — simpler and sufficient at small scale
- Weight tying between token embedding and output projection
- ReLU activation in the MLP block
- Tensor autograd engine (no framework dependency)
- Adam optimizer with linear LR decay and bias correction
- Cross-entropy loss (fused softmax + NLL for numerical stability)
- Training loop with checkpointing, inference loop with temperature sampling

Training targets browser deployment from the outset. The inference path uses WebGPU for matrix operations and WASM for CPU-side orchestration (autograd, Adam, tokenizer, training loop).

### Tensor autograd (`src/tensor.ts`)

WasmGPT uses tensor-based autograd instead of consgpt.lisp's scalar autograd. Each weight matrix is a single `Tensor` object backed by a contiguous `StaticArray<f32>`, so the computation graph has one node per operation instead of one node per scalar. This prepares for future WebGPU acceleration: `StaticArray<f32>` maps directly to GPU buffer uploads, and tensor operations can be swapped for compute shader dispatches without structural changes.

| Scalar (consgpt.lisp) | Tensor (wasmgpt) |
|---|---|
| `Val` per f32 value | `Tensor` per matrix/vector |
| `localGrads: Array<f32>` per node | `op: i32` enum + dispatch function |
| `vadd(a, b)` creates 1 node for 1 scalar | `add(a, b)` creates 1 node for N elements |
| `linear(x, w)` = m*n vmul + m vsum nodes | `matmul(w, x)` = 1 node |
| Millions of nodes per forward pass | Dozens of nodes per forward pass |

**Tensor class:**

```
class Tensor {
  data: StaticArray<f32>;      // contiguous storage, row-major
  grad: StaticArray<f32>;      // same size as data, accumulated during backward
  shape: StaticArray<i32>;     // [rows, cols] for 2D, [len] for 1D, [] for scalar
  children: Array<Tensor>;     // input tensors (empty array for leaves)
  op: i32;                     // which operation produced this tensor
  scalarArg: f32;              // auxiliary scalar (for MUL_SCALAR, DIV_SCALAR, RMSNORM eps)
  intArg: i32;                 // auxiliary int (for EMBEDDING row index, SLICE start)
  intArg2: i32;                // auxiliary int (for SLICE end)
  cacheData: StaticArray<f32>; // auxiliary storage (cross-entropy stores softmax probs here)
}
```

AssemblyScript has no closures, so we cannot store a backward function on each node (as PyTorch does). Instead, each Tensor stores an `op: i32` enum value (17 ops total), and `backward()` dispatches on it:

| Op | Forward | Backward (given output grad `g`) |
|---|---|---|
| `MATMUL` | `C = A @ B` (A is [m,k], B is [k,1]) | `A.grad += g @ B^T`, `B.grad += A^T @ g` |
| `ADD` | `C = A + B` elementwise | `A.grad += g`, `B.grad += g` |
| `RELU` | `C[i] = max(0, A[i])` | `A.grad[i] += g[i] * (A.data[i] > 0 ? 1 : 0)` |
| `SOFTMAX` | `C = softmax(A)` | `A.grad[i] += C.data[i] * (g[i] - dot(g, C.data))` |
| `RMSNORM` | `C = x / rms(x)` | Chain rule through normalization |
| `LOG` | `C[i] = log(A[i])` | `A.grad[i] += g[i] / A.data[i]` |
| `NEG` | `C[i] = -A[i]` | `A.grad[i] += -g[i]` |
| `EMBEDDING` | `C = table[id]` (row lookup) | `table.grad[id*cols..(id+1)*cols] += g` |
| `SUM` | `C = sum(A)` (scalar output) | `A.grad[i] += g[0]` for all i |
| `MUL_SCALAR` | `C[i] = A[i] * s` | `A.grad[i] += g[i] * s` |
| `DIV_SCALAR` | `C[i] = A[i] / s` | `A.grad[i] += g[i] / s` |
| `SLICE` | `C = A[start..end]` | `A.grad[start+i] += g[i]` |
| `CONCAT` | `C = [A₀; A₁; ...; Aₙ]` | `Aₕ.grad += g[hStart..hEnd]` for each part |
| `MUL` | `C[i] = A[i] * B[i]` | `A.grad[i] += g[i] * B.data[i]`, `B.grad[i] += g[i] * A.data[i]` |
| `CROSS_ENTROPY` | `C = -log(softmax(A)[target])` | `A.grad[i] += g[0] * (probs[i] - (i == target ? 1 : 0))`, probs from `cacheData` |
| `SCALE` | `C[i] = A[i] * B.data[0]` (vec * scalar) | `A.grad[i] += g[i] * B.data[0]`, `B.grad[0] += sum(g[i] * A.data[i])` |

**`backward(loss)`** follows the same algorithm as consgpt.lisp's scalar version:
1. Iterative DFS topological sort (using `Set<usize>` for visited tracking)
2. Set `loss.grad[0] = 1.0`
3. Reverse walk: dispatch on `op` to accumulate gradients into children
4. Detach graph: clear `children` on all nodes for GC

### GPT-2 model (`src/model.ts`)

**Hyperparameters:** Stored as private module variables with getter functions (`getNEmbd()`, `getNLayer()`, `getNHead()`, `getHeadDim()`, `getBlockSize()`). Defaults match consgpt.lisp's Tiny config: nEmbd=64, nLayer=2, nHead=4, headDim=16, blockSize=256. At runtime, `train.ts` and `infer.ts` call `setHyperparams()` with values read from `config.sexp`.

**PRNG:** Same LCG as consgpt.lisp — `state = (1664525 * state + 1013904223) & 0xFFFFFFFF`, seeded at 42. Box-Muller transform for Gaussian initialization.

**Weight initialization:** `makeMatrix(nout, nin)` creates a [nout, nin] Tensor filled with Gaussian * `initScale` (configurable, default 0.02).

**State dictionary:** `stateDict: Map<string, Tensor>` with 14 weight tensors:
- `wte`: [vocabSize, nEmbd], `wpe`: [blockSize, nEmbd]
- Per layer (2 layers): `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo` (all [nEmbd, nEmbd]), `mlp_fc1` ([4\*nEmbd, nEmbd]), `mlp_fc2` ([nEmbd, 4\*nEmbd])
- Weight tying: `wte` reused for output projection (no separate `lm_head`)

**Forward pass** (`gpt(tokenId, posId, cacheKeys, cacheVals) → logits`):

```
x = add(embedding(wte, tokenId), embedding(wpe, posId))
x = rmsnorm(x, 1e-5)

for each layer:
  // Multi-head attention with KV cache
  q, k, v = matmul(wq/wk/wv, rmsnorm(x))
  cache k, v
  for each head:
    scores = softmax(concat([dot(qH, kH) / sqrt(headDim) for each cached position]))
    headOut = weighted sum of cached values using scale(vH, attnWeight)
  x = add(matmul(wo, concat(headOuts)), residual)

  // MLP
  x = add(matmul(fc2, relu(matmul(fc1, rmsnorm(x)))), residual)

logits = matmul(wte, x)    // weight tying
```

The attention weighted sum uses `OP_SCALE` (vector * scalar tensor) instead of `mulScalar` (vector * raw f32) to maintain gradient flow through softmax. Dot products use `tensorSum(mul(qH, kH))`.

### Training pipeline (`src/train.ts`)

WASI CLI program: `wasmtime --dir . build/train.wasm <corpus.wat> <merges.sexp> [numSteps]`

**Data loading:** initVocabulary → parseMerges → buildBpeVocab → tokenize corpus → encode tokens to IDs via vocab lookup (Pass 1) or bpeEncodeToken (Pass 2).

**Training configuration:** Read from `config.sexp` at startup via `parseConfig()`, with fallback defaults: `train-seq-len` (256), `learning-rate` (0.001), `beta1` (0.9), `beta2` (0.999), `eps-adam` (1e-8), `checkpoint-interval` (10). Model architecture hyperparameters are also loaded from config and applied via `setHyperparams()` before model initialization.

**Adam optimizer:** Per-tensor `Map<string, StaticArray<f32>>` for first moment (m) and second moment (v). Linear LR decay relative to current run. Bias correction uses absolute step count. Gradients zeroed after update.

**Batch extraction:** `getBatch(step)` returns a contiguous window of `TRAIN_SEQ_LEN + 1` IDs. Start index: `(step * TRAIN_SEQ_LEN) % max(1, corpusLen - TRAIN_SEQ_LEN - 1)`.

**Cross-entropy loss:** Fused `OP_CROSS_ENTROPY` (softmax + NLL in one op) stores softmax probabilities in `cacheData` for backward. More numerically stable and efficient than the composed version `neg(logOp(slice(softmax(logits), target, target+1)))`.

### Inference pipeline (`src/infer.ts`)

WASI CLI program: `wasmtime --dir . build/infer.wasm <vocab.sexp> [numSamples] [temperature] [prompt...]`

**Vocabulary loading:** Inference loads the full vocabulary from `build/vocab.sexp` (written by `train-bpe.ts` after BPE training) rather than initializing `vocabulary.ts` + `bpe.ts`. The S-expression file provides both forward (`tokenToId: Map<string, i32>`) and reverse (`idToToken: Map<i32, string>`) mappings. Vocab size is computed as `max(ID) + 1` across all entries.

**Prompt encoding:** If a prompt is provided, it is tokenized by the lexer and encoded via simple `tokenToId` lookup. Unknown prompt tokens are skipped with a warning. Each prompt token is fed through `gpt()` one at a time, building the KV cache, with graph detachment after each position.

**BOS token:** `<BOS>` is a Pass 1 control token (ID 568) registered in `vocabulary.ts`. Following microgpt.py's approach, the training corpus is wrapped with BOS on both sides (`[BOS] + corpus + [BOS]`), teaching the model that BOS → first token and last token → BOS. Inference starts from BOS and stops when BOS is generated, producing variable-length output. consgpt.lisp registers BOS but does not inject it into training data — we follow microgpt's approach instead because without BOS in training data, the model has no signal to learn when to stop generating.

**Autoregressive generation:** Starting from BOS, the loop runs until the model generates BOS again or reaches `BLOCK_SIZE`:

```
logits = gpt(tokenId, posId, cacheKeys, cacheVals)
scaled = divScalar(logits, temperature)
probs = softmax(scaled)
nextId = weightedChoice(probs.data)
detachKvCache(cacheKeys, cacheVals)
if nextId == BOS_ID: break
```

Temperature (default 0.8) controls sampling randomness — lower values concentrate probability mass on the most likely tokens.

**KV cache detachment:** During training, `backward()` detaches the entire computation graph. During inference there is no backward pass, so the graph grows unboundedly as tokens are generated. After sampling each token, `detachKvCache` walks all cached K and V tensors across all layers and clears their `children` arrays. This preserves tensor data (needed for future attention) while breaking graph references so intermediate nodes can be GC'd.

**Token decoding:** Converts generated IDs back to WAT text via `idToToken` lookup. Spacing heuristics: no space after `(`, no space before `)`, space between all other adjacent tokens. Unknown IDs decode as `<?>`.

**Checkpoint loading:** Inference aborts if the checkpoint is missing or has a config mismatch. It passes dummy Adam m/v maps to `loadCheckpoint` (same-shaped zero arrays, discarded after load) since the checkpoint format includes optimizer state that must be read past.

**Differences from consgpt.lisp inference:**

| Aspect | consgpt.lisp | wasmgpt |
|---|---|---|
| Start token | BOS (after BPE vocab, not in training data) | BOS (Pass 1 ID 568, wrapped in training data) |
| Stop condition | BOS token generated | BOS generated or BLOCK_SIZE reached |
| Temperature | 0.8 (scalar Val division) | 0.8 (tensor divScalar) |
| Decode spacing | Lisp-specific (no space after `(`, `'`, `` ` ``, `#'`, `#(`, `#\\`) | WAT-specific (no space after `(`, no space before `)`) |
| Graph detachment | Nil out children/local-grads on scalar Val nodes | Clear children arrays on Tensor nodes |
| Checkpoint loading | Ignores Adam state | Passes dummy Adam maps, discards after load |
| Prompt support | No (always starts from BOS) | Optional prompt prefix |

### Checkpoint persistence (`src/checkpoint.ts`)

Binary format (`build/model.bin`) — compact, natural for WASM, trivial to read/write via linear memory:

```
[i32]  step number
[i32]  nEmbd
[i32]  nLayer
[i32]  nHead
[i32]  blockSize
[i32]  vocabSize
[f32 x N]  weight values (all tensors in sorted key order, row-major)
[f32 x N]  Adam first moment (m) — same order
[f32 x N]  Adam second moment (v) — same order
```

Uses raw WASI `fd_write`/`fd_read` with `changetype<usize>(staticArray)` for zero-copy binary I/O. On load, hyperparameters are validated against the current model config. The file is overwritten on each save.

### Configuration (`config.sexp`, `src/config.ts`)

All tunable hyperparameters are stored in `config.sexp`, an S-expression file in the project root. Both `train.ts` and `infer.ts` load it at startup via `parseConfig()`, which reuses the existing WAT lexer to tokenize the file and extracts `(key value)` pairs into a `Map<string, f64>`. Typed accessors `configI32()` and `configF32()` retrieve values with fallback defaults.

Parameters include model architecture (n-embd, n-layer, n-head, block-size, init-scale), training (train-seq-len, learning-rate, beta1, beta2, eps-adam, checkpoint-interval), and inference (temperature). Each parameter in the file has a comment explaining its purpose and the corresponding GPT-2 value for reference.

The model's default hyperparameters (in `model.ts`) serve as the single source of truth for fallback values — `train.ts` and `infer.ts` pass getter results (e.g. `getNEmbd()`) as defaults to `configI32()`/`configF32()`, avoiding hardcoded literals in multiple places.

### Module dependency order

```
lexer.ts
  └─ vocabulary.ts
  └─ bpe.ts (imports lexer for S-expression parsing)
  └─ config.ts (imports lexer for S-expression parsing)
  └─ train-bpe.ts
io.ts (safe file reading, bypasses as-wasi readString bug)
tensor.ts (independent)
model.ts (depends on tensor)
checkpoint.ts (depends on model)
train.ts (depends on all above + io.ts + config.ts)
infer.ts (depends on model, checkpoint, bpe, io.ts, config.ts)
```

---

## Tokenization

### Design Philosophy

WASM has two classes of content that require different tokenization strategies:

**Structured content** (fixed vocabulary):
- WAT instruction mnemonics (e.g. `i32.add`, `local.get`, `call`)
- WAT binary opcodes (e.g. `0x6A`)
- These map to the same token ID regardless of representation format

**Unstructured content** (open-ended vocabulary):
- Immediate values (e.g. `42` in `i32.const 42`)
- Data section contents
- Comments
- Function signatures, type annotations, identifiers
- Imported function names (including WebGPU API calls)

### Hybrid Tokenizer

A hybrid tokenizer handles both classes, following the same two-pass design as [consgpt.lisp](https://github.com/spratt/consgpt.lisp):

1. **Pass 1: Instruction tokens** (`src/vocabulary.ts`) — analogous to consgpt.lisp's `symbol-tokenizer.lisp`. Registers all known tokens with fixed integer IDs via an idempotent `reg()` function. Registration order:
   1. Syntax tokens: `(`, `)`
   2. Structural WAT keywords: `module`, `type`, `func`, `import`, `export`, `memory`, `table`, `global`, `data`, `elem`, `start`, `param`, `result`, `local`, `mut`, `offset`, `align`
   3. Type names: `i32`, `i64`, `f32`, `f64`, `v128`, `funcref`, `externref`
   4. All 542 WASM instruction mnemonics from [wabt's `opcode.def`](https://github.com/WebAssembly/wabt), grouped by category: control flow, variable access, reference types, table ops, memory loads/stores, memory ops, bulk memory, constants, i32/i64/f32/f64 numeric, conversions, i32/i64 atomics, atomic fence/wait/notify, SIMD v128, SIMD i8x16/i16x8/i32x4/i64x2/f32x4/f64x2

   568 tokens are registered; after deduplication (15 structural keywords overlap with instruction mnemonics, e.g. `block`, `loop`, `if`, `call`, `select`), **~553 unique tokens** remain. The `reg()` function is idempotent — registering a token that already exists is a no-op.

   Mnemonics were extracted from wabt's `opcode.def` using:
   ```bash
   grep '^WABT_OPCODE' opcode.def | grep -v 'Interp' | awk -F'"' '{print $2}' | sort -u
   ```
   Interpreter-only opcodes (marked `Interp`) are excluded. The only duplicate text mnemonic is `select` (appears for both `Select` and `SelectT` enum entries).

   **Exports:** `vocab: Map<string, i32>`, `nextId: i32`, `initVocabulary(): void`

2. **Pass 2: BPE tokens** (`src/bpe.ts`) — ported from consgpt.lisp's `bpe.lisp`. The BPE algorithm is language-agnostic. Everything not in the Pass 1 vocabulary (immediates, identifiers, comments, data) is tokenized via Byte Pair Encoding trained on the corpus.

   **Training:** `trainBpe(unknownTokens, numMerges)` builds a frequency table of character-split words, then iteratively finds and applies the most frequent adjacent pair as a merge rule.

   **Encoding:** `bpeEncodeToken(token)` splits the token into characters, applies all learned merges via `applyMerges()`, then maps each resulting subword to its integer ID. Unknown subwords fall back to `<UNK>`.

   **Vocabulary building:** `buildBpeVocab(merges, startId)` assigns IDs starting from Pass 1's `nextId`, registering single characters (sorted) first, then merged symbols in merge order, then `<UNK>` as the final token.

   **Key encoding:** Since AssemblyScript Maps require string keys, word arrays are encoded as `\x01`-separated strings and pair keys use `\0` as the separator (`left + "\0" + right`).

   **Exports:** `bpeMerges`, `bpeVocab`, `unkId`, `tokenToChars()`, `mergePair()`, `countPairs()`, `trainBpe()`, `applyMerges()`, `bpeEncodeToken()`, `buildBpeVocab()`, `getBpeNextId()`, `serializeMerges()`, `parseMerges()`, `serializeVocab()`, `parseVocab()`, `escapeForSexp()`, `unescapeFromSexp()`, `stripQuotes()`

### BPE Training Pipeline

BPE merge rules must be trained on a corpus before model training can begin. The pipeline is:

1. **Build** — compile wasmgpt's AssemblyScript source to WASM (`npm run build`)
2. **Disassemble** — convert to WAT with offset map (`npm run wat`, requires `wasm2wat` on PATH)
3. **Annotate** — inject source comments via [watnot](https://github.com/spratt/watnot) (`npm run annotate`, watnot is a git submodule)
4. **Train** — run `src/train-bpe.ts` on the annotated WAT to learn merge rules (`npm run train:bpe`)

The full pipeline is chained as `npm run train:bpe`, which runs all four steps and writes merge rules to `build/merges.sexp` and vocabulary to `build/vocab.sexp`.

`src/train-bpe.ts` is a WASI CLI program (following watnot's I/O pattern) that:
- Reads a WAT file path from CLI args
- Tokenizes with the lexer, partitions tokens into known (in `vocab`) and unknown
- Calls `trainBpe(unknowns, numMerges)` (default 256 merges)
- Writes merge rules to `build/merges.sexp` and vocabulary to `build/vocab.sexp`
- Writes diagnostic statistics to stderr

**Persistence format:** S-expressions with quoted strings and backslash escaping. Merges: `(merges ("left" "right") ...)`. Vocabulary: `(vocab ("token" id) ...)`. Parsed by the existing WAT lexer via `parseMerges()` and `parseVocab()`, serialized by `serializeMerges()` and `serializeVocab()`. S-expressions were chosen over TSV because BPE tokens can contain `\0`, `\n`, and `\t` which corrupt delimiter-based formats.

**Initial corpus statistics** (self-referential build of wasmgpt):
- 11,849 lines of annotated WAT
- 53,102 total tokens after lexing
- 42,187 known tokens (79.4%, in Pass 1 vocabulary)
- 10,915 unknown tokens (20.6%, encoded by BPE)
- 256 merge rules learned

Comments are stripped by the lexer, so watnot's injected comments don't affect tokenization. The annotations are preserved in the corpus for future use when comment-aware training is implemented.

### Format Agnosticism

The tokenizer operates in two modes:
- **WAT mode** — parses text, recognizes instruction mnemonics and identifiers
- **Binary mode** — reads WASM binary, recognizes opcodes and LEB128-encoded immediates

Both modes emit the same token IDs for instructions, making the model format-agnostic. A mixed corpus of WAT and WASM binary is valid input.

### WebGPU Calls

WebGPU calls appear in WASM as imported function calls — a `call` instruction with a function index pointing to a host import. These are handled by the BPE side of the tokenizer. Optionally, well-known WebGPU API functions may be added as dedicated tokens if the corpus contains sufficient WebGPU-heavy WASM.

---

## Corpus Generation

### Sources

The training corpus is assembled from multiple sources:

1. **Real WASM binaries from the web** — disassembled to WAT via `wasm2wat`
2. **WAT source files** — collected directly from open source repositories
3. **AssemblyScript source** — compiled to WAT with source maps, then annotated (see below)
4. **Project self-reference** — the source code of this project itself, compiled to annotated WAT, is included in the corpus

### Comment Transplantation Pipeline

The highest-quality corpus entries are produced by transplanting human-authored comments from AssemblyScript source into the corresponding WAT output. This provides genuine programmer intent as natural language annotations at the instruction level.

Pipeline (implemented by [watnot](https://github.com/spratt/watnot)):

1. Compile AssemblyScript source with `--debug --sourceMap` → WASM binary with source map
2. Parse the original source files to extract all comments with their line numbers
3. Disassemble binary to WAT via `wasm2wat --fold-exprs --offset-map` → folded WAT + offset map (WAT line → WASM byte offset). Requires [our fork of wabt](https://github.com/spratt/wabt/tree/byte_offsets) for `--offset-map` support.
4. Parse the source map to build a mapping: WASM byte offset → (source file, line number)
5. Correlate WAT lines with source locations via the offset map and source map; inject source comments at the corresponding WAT locations

**Example output:**

```wat
(func $allocate (param $size i32) (result i32)
  ;; check if heap has enough space remaining
  (i32.le_u
    (i32.add
      (global.get $heap_ptr)
      ;; align to 8-byte boundary before allocating
      (i32.and
        (i32.add
          (local.get $size)
          (i32.const 7))
        (i32.const -8)))
    (global.get $heap_end)))
```

Comments are entirely human-authored and reflect genuine programmer intent — a higher quality training signal than synthetically generated annotations.

### Why AssemblyScript

AssemblyScript is the preferred implementation and corpus generation language because:
- Designed specifically for WASM as its target; output WAT is significantly more readable than Rust or C
- Function names are preserved in output WAT
- Local variable names are preserved
- Source map output is supported via `--sourceMap` flag
- TypeScript-like syntax is familiar to web developers

---

## Deployment

The trained model runs inference entirely in the browser:

- **WebGPU** — matrix multiplications (forward pass, attention)
- **WASM** — CPU-side orchestration: tokenizer, KV cache management, sampling
- **No server required** — weights are loaded client-side and cached

This mirrors the WebLLM architecture but for a domain-specific model trained from scratch rather than a quantized general-purpose model.

---

## Future Phases

### Natural Language Interface

The model as described only understands WAT — it cannot respond to natural language prompts. A future phase adds this capability:

1. Use a general-purpose LLM (e.g. Claude or GPT-4) to generate natural language descriptions of WAT functions in the corpus
2. Produce paired examples: (natural language description → WAT implementation)
3. Fine-tune the model on these pairs

This enables prompt-to-WASM generation without requiring natural language to be present in the original training corpus.

### Distributed Training

The browser deployment target opens the possibility of distributed federated training — browser nodes each run the forward pass, compute gradients via autograd, and report updates to a central aggregator. Fine-tuning is more feasible than training from scratch in this setting due to communication overhead.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Template project | [consgpt.lisp](https://github.com/spratt/consgpt.lisp) | Proven GPT on S-expression language; hybrid tokenizer, autograd, model, training loop all port directly |
| Model architecture | GPT-2 style transformer with RMSNorm and weight tying | Proven in consgpt.lisp at small scale (203K params, Tiny config) |
| Implementation language | AssemblyScript | Readable WAT output, name preservation, source map support |
| Autograd | Tensor-based (op enum dispatch) | 7-8x faster than scalar autograd; `StaticArray<f32>` maps directly to GPU buffers for future WebGPU |
| Tokenizer | Two-pass hybrid (instruction tokens + BPE) | Same design as consgpt.lisp; exploits known WASM structure; BPE handles open-ended values |
| Corpus annotation | Source-map-based comment transplantation via [watnot](https://github.com/spratt/watnot) | Human-authored intent without synthetic generation |
| Deployment | Browser (WASM + WebGPU) | Aligns with domain; no server infrastructure required |
| Training trigger | WAT code completion | Keeps model simple; natural language deferred to fine-tuning phase |
| BPE/vocab persistence | S-expressions with quoted strings | Parsed by existing WAT lexer; safe for tokens containing `\0`, `\n`, `\t` |
| Checkpoint format | Binary (raw WASI fd_write/fd_read) | Zero-copy I/O via `changetype<usize>(staticArray)`; compact and natural for WASM |
| Self-reference | Project source in corpus | Elegant bootstrapping; increases corpus coverage of idiomatic WAT |
| Configuration | S-expression file (`config.sexp`) parsed by WAT lexer | All hyperparameters tunable without recompilation; reuses existing lexer; comments document GPT-2 reference values |

---

## AssemblyScript Constraints

Key constraints encountered during implementation:

- **No closures** — all functions must be top-level or class methods. This is why tensor autograd uses op enum dispatch instead of stored backward functions (as PyTorch does).
- **No `for..of`** — use index-based loops.
- **Field initializers required** on classes.
- **Sort comparators** must be top-level functions.
- **`<` after `.get()`** is interpreted as a generic type parameter — store `.get()` result in a local variable first.
- **`f32` arithmetic** may need explicit `f32()` casts to avoid implicit promotion to f64.
- **`Map` keys** must be value types or strings — no object keys.
- **`StaticArray<f32>`** for contiguous tensor storage (not `Array<f32>` — StaticArray has fixed size and no indirection, maps cleanly to linear memory and future GPU buffers).
- **`Mathf.*`** for f32 math (`Mathf.log`, `Mathf.exp`, `Mathf.sqrt`, `Mathf.pow`, `Mathf.max`, `Mathf.abs`), not `Math.*` which returns f64.
- **Iterative DFS** for topological sort — recursive DFS may hit stack limits on deep graphs.
- **`Set<usize>`** works for visited tracking — `changetype<usize>(tensor)` gives the unique object address.
- **`changetype<usize>(staticArray)`** gives the memory address of a StaticArray's data — used for efficient binary I/O in checkpoint persistence.

### Implementation lessons

- The **`crossEntropy` cacheData pattern** — storing softmax probabilities in `cacheData` during forward for use in backward — works well as a closure substitute. Reusable for any op that needs intermediate results during backward.
- The **op enum dispatch pattern** is clean and extensible: adding a new op requires (1) a constant, (2) a forward function, (3) a backward case in `backwardOp`.
- **`rmsnorm` backward** has a non-trivial chain rule. The clean formula is `a.grad[i] += rmsScale * (g[i] - t.data[i] * dotGOut / n)`.
- **`concat` backward** must index into the output grad with offset (`g[offset + j]`) but into each child's grad from 0 (`child.grad[j]`).
- **`export let` is not a live binding** — `export let nextId` in vocabulary.ts was imported by other modules, but AssemblyScript captures the value at import time (unlike ES module live bindings). After `initVocabulary()` set `nextId = 568`, importers still saw 0. Fixed by replacing with a `getNextId()` getter function.
- **as-wasi `readString()` / `readAll()` corrupts large files** — both use `memory.data()` for iov and read_ptr buffers. These are static memory addresses that get clobbered when other parts of the program (Console.error, other fd_read calls) use `memory.data()` with the same size. In a minimal program the corruption doesn't occur; in a larger program (train.ts with vocabulary init, multiple file reads, stderr output) the static buffers are overwritten mid-read, producing null bytes after ~2048 bytes. A debug harness proved: `readString()` → 2047 null bytes, 0 merges parsed; `readFileText()` (heap-allocated buffers) → 0 null bytes, 256 merges. Fixed by creating `src/io.ts` with `readFileText()` that calls `fd_read` directly using heap-allocated `ArrayBuffer`s for iov and read_ptr.
- **TSV format cannot safely represent arbitrary tokens** — BPE merge rules can contain tokens like `\0`, `\n`, `\t`, and `\`. TSV uses `\t` and `\n` as delimiters, so these tokens corrupt the format. `parseMerges` read only 135 of 256 merge rules from the TSV file because `split("\n")` misinterpreted byte sequences within tokens. Fixed by switching to S-expression serialization with quoted strings and backslash escaping, parsed by the existing WAT lexer.
- **Vocab size must be computed consistently** — `buildBpeVocab` reuses Pass 1 IDs for merged symbols that match known tokens (e.g., `i32`, `offset`). This means `bpeNextId` (sequential counter) can differ from `max(ID) + 1` across all vocab entries. Training used `getBpeNextId()` while inference used `max(ID) + 1` from the vocab file, causing a checkpoint config mismatch (return code -2). Both now use `getBpeNextId()` = 876.

---

## Open Questions

- What vocabulary size is appropriate for the BPE component given the expected corpus size?
- Should the model be trained on WAT, binary, or a mix? (Binary gives more data volume; WAT gives better annotation quality)
- How should the tokenizer handle WASM 3.0 extensions vs. MVP instructions — unified vocabulary or versioned?
- What is a reasonable model size given browser memory constraints for inference?
- How to handle the structured block nesting in WAT (blocks must be well-formed) — should this be enforced at the tokenizer level or left to the model to learn?
- Should we support top-k or top-p (nucleus) sampling? Temperature-only is simpler and matches consgpt.lisp.
