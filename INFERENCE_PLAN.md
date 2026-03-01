# Plan: Inference Pipeline

## Context

The training pipeline is complete. We have a trained checkpoint at `build/model.bin` containing weights and optimizer state. Inference loads the checkpoint, runs the model autoregressively, and generates WAT code via temperature-scaled sampling.

The reference implementation is `consgpt.lisp/infer.lisp`. The inference pipeline is simpler than training: no backward pass, no optimizer, no loss computation. The main challenges are KV cache memory management (graph detachment without backward) and token decoding (ID → string, which requires a reverse vocabulary mapping that doesn't currently exist).

## Step 0: Research findings

- No reverse vocabulary mapping exists — `vocab` and `bpeVocab` are both `Map<string, i32>` (forward only). Rather than building the reverse map in memory (wastes memory during training) or computing it at inference startup (wastes compute), we serialize the full vocabulary to `build/vocab.sexp` during the BPE training step. Inference loads this TSV to get both forward (`token → ID`) and reverse (`ID → token`) mappings without initializing vocabulary.ts or bpe.ts.
- `divScalar(a, s)` already exists in tensor.ts — creates a graph node but that's fine for inference (graph gets detached per step).
- `softmax(a)` already exists in tensor.ts — numerically stable with max subtraction.
- No explicit single-tensor `detach()` function — we need a `detachKvCache` helper that walks all cached K/V tensors and clears their `children` arrays.
- `loadCheckpoint` requires Adam m/v maps — inference passes dummy zero-initialized maps (same pattern as train.ts lines 110-119), discards them after load.
- No BOS/EOS token in wasmgpt — generation starts from `(` (the most common WAT opening token, ID 0) or a user-provided prompt, and runs until BLOCK_SIZE.
- `bpeVocab` includes single characters as entries (not just merged pairs), so the reverse map covers all BPE token IDs.
- Clean ID handoff: Pass 1 IDs are 0..nextId-1, BPE IDs start at nextId with no gap.

## What already exists

- `gpt(tokenId, posId, cacheKeys, cacheVals) → logits` — forward pass (model.ts)
- `softmax(a) → Tensor` — numerically stable softmax (tensor.ts)
- `loadCheckpoint(path, adamM, adamV) → step` — loads weights from binary checkpoint (checkpoint.ts)
- `initModel(vocabSize)` — allocates weight tensors (model.ts)
- `initVocabulary()` — populates Pass 1 vocab (vocabulary.ts)
- `parseMerges()`, `buildBpeVocab()` — loads BPE vocabulary (bpe.ts)
- `randomUniform()` — LCG PRNG (model.ts)
- `vocab: Map<string, i32>` — token → ID (vocabulary.ts)
- `bpeVocab: Map<string, i32>` — subword → ID (bpe.ts)

## What needs to be built

### 1. Vocabulary serialization and reverse mapping

Currently only forward maps exist (`token → ID`). Inference needs `ID → token` for decoding generated sequences.

**Approach:** Serialize the full vocabulary (Pass 1 + BPE) to `build/vocab.sexp` during BPE training, after `buildBpeVocab` completes. Each line is `id\ttoken`. This follows the same pattern as `build/merges.sexp` for BPE merge rules.

At inference time, load `build/vocab.sexp` to build both:
- `tokenToId: Map<string, i32>` — for encoding prompts
- `idToToken: Map<i32, string>` — for decoding generated tokens

This avoids carrying an unused reverse map during training (memory) and avoids recomputing the full vocabulary at inference startup (compute). The TSV file is the single source of truth for the vocabulary at inference time.

### 2. Token decoding with spacing heuristics

Convert an array of token IDs back to readable WAT text. Spacing heuristics for WAT differ from Lisp:

- **No space after:** `(`
- **No space before:** `)`
- **All other tokens:** space-separated

This is simpler than consgpt.lisp's decode because WAT has fewer special syntax characters.

### 3. Temperature sampling

Scale logits by temperature before softmax, then sample from the resulting probability distribution:

```
scaled[i] = logits[i] / temperature
probs = softmax(scaled)
nextId = weightedChoice(probs)
```

`weightedChoice` generates a random threshold in `[0, 1)` and walks the probability array accumulating mass until the threshold is reached.

Temperature 0.8 is the consgpt.lisp default — slightly more deterministic than uniform sampling.

### 4. KV cache detachment

During training, `backward()` detaches the entire computation graph. During inference there is no backward pass, so the graph grows unboundedly as tokens are generated. After sampling each token, we must explicitly detach the KV cache tensors:

```
for each layer:
  for each cached K tensor: clear children
  for each cached V tensor: clear children
```

This preserves the tensor data (needed for future attention computations) while breaking graph references so intermediate nodes can be GC'd.

### 5. Weights-only checkpoint loading

`loadCheckpoint` currently requires Adam m/v maps as parameters and reads them from the file. For inference we don't need optimizer state. Two options:

**Option A:** Pass dummy m/v maps (same size as weights, discarded after load). Simple, no changes to checkpoint.ts.

**Option B:** Add a `loadCheckpointWeightsOnly` function that reads the header and weights but seeks past Adam state. Cleaner but requires new code.

Option A is simpler and sufficient. The Adam arrays are small relative to the weights and are immediately discardable.

### 6. Prompt support (optional)

consgpt.lisp starts generation from a BOS token. For WAT, a more useful mode is **prompt completion** — the user provides a WAT prefix (e.g., `(module (func`), it gets tokenized and fed through the model position by position, then generation continues autoregressively from where the prompt left off.

This is a natural extension: encode the prompt tokens, feed them through `gpt()` one at a time (building the KV cache), then switch to sampling mode.

## Implementation steps

### Step 1: Add vocabulary serialization to BPE training pipeline

After `buildBpeVocab` completes in `src/train-bpe.ts`, write `build/vocab.sexp` to stdout (or a separate file). Each line: `id\ttoken`. Covers all entries from both `vocab` and `bpeVocab`.

Alternatively, add a `serializeVocab()` function to `src/bpe.ts` (alongside `serializeMerges()`) that iterates both maps and produces the TSV. The train-bpe entry point calls it and writes the output.

**Files:** Edit `src/bpe.ts` (add `serializeVocab`), edit `src/train-bpe.ts` (write vocab.sexp)

### Step 2: Add `weightedChoice` to model.ts

```
function weightedChoice(probs: StaticArray<f32>): i32
```

Uses `randomUniform()` (already exported from model.ts) to generate a threshold, walks the probability array accumulating mass, returns the index where cumulative probability exceeds the threshold.

This belongs in model.ts alongside the existing PRNG functions.

**Files:** Edit `src/model.ts`

### Step 3: Add `detachKvCache` helper

```
function detachKvCache(
  cacheKeys: Array<Array<Tensor>>,
  cacheVals: Array<Array<Tensor>>
): void
```

Walks all cached K and V tensors across all layers, clears their `children` arrays. This breaks the computation graph while preserving tensor data for future attention.

This can live in model.ts (near `gpt()`) or in the inference entry point.

**Files:** Edit `src/model.ts` or create in `src/infer.ts`

### Step 4: Create `src/infer.ts` — inference entry point

WASI CLI program:

```
wasmtime --dir . build/infer.wasm <vocab.sexp> [numSamples] [temperature] [prompt...]
```

**Flow:**

1. Parse CLI args (vocab path, optional numSamples default 5, temperature default 0.8, optional prompt words joined with spaces)
2. Load vocabulary from `build/vocab.sexp`: parse each `id\ttoken` line, build both `tokenToId: Map<string, i32>` and `idToToken: Map<i32, string>`, track max ID for vocab size
3. Initialize model with `initModel(vocabSize)`
4. Load checkpoint with dummy Adam maps (Option A), discard after load
5. For each sample:
   a. Initialize fresh KV cache (empty arrays, one per layer)
   b. If prompt provided: tokenize prompt (using the lexer), encode tokens via `tokenToId` lookup, feed through `gpt()` position by position (building KV cache), detach after each position
   c. Autoregressive generation loop up to `BLOCK_SIZE`:
      - `logits = gpt(tokenId, posId, cacheKeys, cacheVals)`
      - `scaled = divScalar(logits, temperature)`
      - `probs = softmax(scaled)`
      - `nextId = weightedChoice(probs.data)`
      - `detachKvCache(cacheKeys, cacheVals)`
      - Collect nextId
      - Stop if sequence reaches BLOCK_SIZE
   d. Decode token IDs to string with spacing heuristics
   e. Print generated text to stdout

**Decode function:**

```
function decodeIds(ids: Array<i32>, idToToken: Map<i32, string>): string
```

Maps each ID through `idToToken`, falls back to `"<?>"` for unknown IDs. Applies spacing: no space after `(`, no space before `)`, space between all other adjacent tokens.

**Note on prompt encoding:** Prompt tokens are encoded via simple `tokenToId` lookup (loaded from vocab.sexp). Unknown tokens in the prompt are skipped with a warning. This is simpler than full BPE encoding but sufficient for WAT prompts where most tokens are known mnemonics/syntax. Full BPE encoding of prompt tokens (splitting unknowns into subwords) is a future enhancement.

**Files:** Create `src/infer.ts`, edit `package.json` (add `build:infer` and `infer` scripts), edit `as-test.config.js` (exclude `src/infer.ts`)

### Step 5: Tests

Test the new components:

- `weightedChoice`: given uniform probabilities, returns values in valid range; given one-hot probabilities, always returns that index
- `decodeIds`: correct spacing for `( module ( func ) )` → `(module (func))`
- `detachKvCache`: after detach, all cached tensors have empty children arrays

These can go in `tests/model.test.ts` (for weightedChoice) and a new `tests/infer.test.ts` (for decode and detach).

**Files:** Edit `tests/model.test.ts`, create `tests/infer.test.ts`

### Step 6: Verification

1. `npm test` — all tests pass
2. `npm run build:infer` — compiles
3. Run inference: `npm run infer` — generates WAT text from trained checkpoint
4. Verify output is plausible WAT (parentheses roughly balanced, recognizable mnemonics)

## Package.json scripts

```json
"build:infer": "asc src/infer.ts --outFile build/infer.wasm --config node_modules/@assemblyscript/wasi-shim/asconfig.json --debug",
"infer": "npm run build:infer && wasmtime --dir . build/infer.wasm build/vocab.sexp"
```

## Key differences from consgpt.lisp inference

| Aspect | consgpt.lisp | wasmgpt |
|---|---|---|
| Start token | BOS (special token after BPE vocab) | First token of prompt, or `(` if no prompt |
| Stop condition | BOS token generated | BLOCK_SIZE reached (no BOS token in wasmgpt) |
| Temperature | 0.8 (scalar Val division) | 0.8 (tensor divScalar) |
| Decode spacing | Lisp-specific (no space after `(`, `'`, `` ` ``, `#'`, `#(`, `#\\`; no space before `)`) | WAT-specific (no space after `(`; no space before `)`) |
| Graph detachment | Nil out children/local-grads on scalar Val nodes | Clear children arrays on Tensor nodes |
| Checkpoint loading | Ignores Adam state via `(declare (ignore adam-m adam-v))` | Passes dummy Adam maps, discards after load |
| Prompt support | No (always starts from BOS) | Optional prompt prefix |

## Open questions

- Should we add a BOS/EOS token to wasmgpt's vocabulary? consgpt.lisp uses BOS to signal both start and end of generation. Without it, inference runs until BLOCK_SIZE. Adding one would require retraining.
- Should we support top-k or top-p (nucleus) sampling? Temperature-only is simpler and matches consgpt.lisp.

## Resolved questions

- Temperature is configurable via CLI arg (default 0.8).

## Lessons learned

### 1. AssemblyScript `export let` is not a live binding

`export let nextId` in vocabulary.ts was imported by other modules, but
AssemblyScript captures the value at import time (unlike ES module live
bindings). After `initVocabulary()` set `nextId = 568`, importers still
saw 0. Fixed by replacing with a `getNextId()` getter function.

### 2. as-wasi `readString()` / `readAll()` corrupts large files

`Descriptor.readString()` and `readAll()` use `memory.data()` for iov
and read_ptr buffers. These are static memory addresses that get
clobbered when other parts of the program (e.g., Console.error, other
fd_read calls) use `memory.data()` with the same size. In a minimal
program (debug harness), the corruption doesn't occur. In a larger
program (train.ts with vocabulary init, multiple file reads, stderr
output), the static buffers are overwritten mid-read, producing null
bytes after ~2048 bytes.

**Proof:** A debug harness reading `build/merges.sexp` (4740 bytes)
showed:

| Method | Null bytes | Merges parsed |
|--------|------------|---------------|
| `readString()` | 2047 | 0 |
| `readFileText()` (heap buffers) | 0 | 256 |
| `readAll()` raw bytes (in isolation) | 0 | n/a |

Fixed by creating `src/io.ts` with `readFileText()` that calls `fd_read`
directly using heap-allocated `ArrayBuffer`s for iov and read_ptr,
avoiding the shared `memory.data()` addresses entirely.

### 3. TSV format cannot safely represent arbitrary tokens

BPE merge rules can contain tokens like `\0`, `\00`, `\n`, `\t`, `"`,
and `\`. TSV uses `\t` and `\n` as delimiters, so these tokens corrupt
the format. `parseMerges` read only 135 of 256 merge rules from the TSV
file because `split("\n")` misinterpreted byte sequences within tokens.

Fixed by switching to S-expression serialization:
- Merges: `(merges ("left" "right") ...)`
- Vocab: `(vocab ("token" id) ...)`

Tokens are quoted strings with `\` → `\\` and `"` → `\"` escaping.
The existing WAT lexer parses this format correctly.

### 4. Vocab size must be computed consistently

`buildBpeVocab` reuses Pass 1 IDs for merged symbols that match known
tokens (e.g., `i32`, `offset`). This means `bpeNextId` (sequential
counter) can differ from `max(ID) + 1` across all entries. Training
used `getBpeNextId()` while inference used `max(ID) + 1` from the vocab
file, causing a mismatch that made `loadCheckpoint` reject the model
(config mismatch, return code -2). Both now use `getBpeNextId()` = 876.
