# Plan: Training Pipeline (Tensor-based)

## Context

The tokenizer (lexer + vocabulary + BPE) converts WAT source text into a flat array of integer token IDs. The training pipeline takes that array and trains a GPT-2 model on it. Inference is deferred — it consumes a trained checkpoint but doesn't need to exist for training to work.

The pipeline follows consgpt.lisp's structure, ported to AssemblyScript, but replaces scalar autograd (`Val` per f32) with tensor-based autograd. Each weight matrix is a single `Tensor` object backed by a contiguous `StaticArray<f32>`, so the computation graph has one node per operation instead of one node per scalar. This prepares for future WebGPU acceleration: `StaticArray<f32>` maps directly to GPU buffer uploads, and tensor operations can be swapped for compute shader dispatches without structural changes.

## Step 0: Research findings

### consgpt.lisp reference (scalar autograd)

- `Val` struct: `data` (double-float), `grad` (double-float), `children` (list of Val), `local-grads` (list of double-float)
- Operations: `vadd`, `vmul`, `vpow`, `vlog`, `vexp`, `vrelu`, `vneg`, `vdiv`, `vsum`
- `backward`: DFS topological sort, reverse walk accumulating `local-grad * parent-grad`, then detach graph (nil out children/local-grads for GC)
- `linear(x, w)`: `(mapcar (lambda (wo) (vsum (mapcar #'vmul wo x))) w)` — one Val node per multiply and per sum, thousands of nodes per linear layer
- State dict: `Map<string, list-of-lists-of-Val>` — 2D nested lists of scalar Val nodes
- `*params*`: flattened vector of all Val nodes for optimizer indexing

### Why tensors

A 64x64 weight matrix is 4096 Val objects in scalar autograd. A single `linear` call on a 64-dim vector creates 64x64 vmul nodes + 64 vsum nodes = 4160 intermediate nodes. Two transformer layers with 6 weight matrices each, plus embeddings, plus the forward pass through a sequence of 32 tokens: the graph easily reaches millions of nodes. Each node is a heap-allocated object with children/localGrads arrays.

With tensors, the same 64x64 weight is one Tensor object. `matmul(w, x)` creates one graph node. The graph for a full forward pass has dozens of nodes instead of millions.

### Tensor backward dispatch

AssemblyScript has no closures, so we cannot store a backward function on each node (as PyTorch does). Instead, each Tensor stores an `op: i32` enum value, and `backward()` dispatches on it:

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
| `CONCAT` | `C = [A₀; A₁; ...; Aₙ]` | `Aₕ.grad += g[hStart..hEnd]` for each head |
| `MUL` | `C[i] = A[i] * B[i]` | `A.grad[i] += g[i] * B.data[i]`, `B.grad[i] += g[i] * A.data[i]` |
| `CROSS_ENTROPY` | `C = -log(softmax(A)[target])` | `A.grad[i] += g[0] * (probs[i] - (i == target ? 1 : 0))`, probs from `cacheData` |
| `SCALE` | `C[i] = A[i] * B.data[0]` (A is vector, B is scalar) | `A.grad[i] += g[i] * B.data[0]`, `B.grad[0] += sum(g[i] * A.data[i])` |

### Adam optimizer (from consgpt.lisp train.lisp)

- Config: `trainSeqLen=32`, `numSteps=100`, `lr=0.001`, `beta1=0.9`, `beta2=0.999`, `epsAdam=1e-8`, `checkpointInterval=10`
- Adam m and v: one value per parameter, initialized to 0
- Linear LR decay relative to current run: `lr_t = lr * (1 - i/numSteps)`
- Per-parameter update:
  - `m[j] = beta1 * m[j] + (1 - beta1) * grad[j]`
  - `v[j] = beta2 * v[j] + (1 - beta2) * grad[j]^2`
  - Bias correction uses **absolute step count**: `m_hat = m[j] / (1 - beta1^(step+1))`, `v_hat = v[j] / (1 - beta2^(step+1))`
  - `param[j] -= lr_t * m_hat / (sqrt(v_hat) + eps)`
- Gradients zeroed after update
- Checkpoint saved every `checkpointInterval` steps, plus final if not aligned

## Step 1: `src/tensor.ts` — Tensor autograd

Domain-independent tensor computation graph with automatic differentiation.

### Op enum

```
const OP_NONE: i32 = 0;       // leaf (weights, inputs)
const OP_MATMUL: i32 = 1;
const OP_ADD: i32 = 2;
const OP_RELU: i32 = 3;
const OP_SOFTMAX: i32 = 4;
const OP_RMSNORM: i32 = 5;
const OP_LOG: i32 = 6;
const OP_NEG: i32 = 7;
const OP_EMBEDDING: i32 = 8;
const OP_SUM: i32 = 9;
const OP_MUL_SCALAR: i32 = 10;
const OP_DIV_SCALAR: i32 = 11;
const OP_SLICE: i32 = 12;
const OP_CONCAT: i32 = 13;
const OP_MUL: i32 = 14;          // elementwise multiply (for attention dot product)
const OP_CROSS_ENTROPY: i32 = 15; // fused softmax + negative log-likelihood
const OP_SCALE: i32 = 16;        // vector * scalar tensor (for attention weighted sum)
```

### Tensor class

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

### Operations (all return new Tensor with children and op set)

- **`matmul(a: Tensor, b: Tensor): Tensor`** — matrix-vector multiply: a is [m,k], b is [k] or [k,1], result is [m]
- **`add(a: Tensor, b: Tensor): Tensor`** — elementwise addition, same shape
- **`relu(a: Tensor): Tensor`** — elementwise max(0, x)
- **`softmax(a: Tensor): Tensor`** — vector softmax with max subtraction for stability
- **`rmsnorm(a: Tensor, eps: f32): Tensor`** — root-mean-square normalization
- **`logOp(a: Tensor): Tensor`** — elementwise natural log (name avoids AS keyword conflict)
- **`neg(a: Tensor): Tensor`** — elementwise negation
- **`embedding(table: Tensor, id: i32): Tensor`** — extract row `id` from [vocabSize, nEmbd] table
- **`sum(a: Tensor): Tensor`** — reduce to scalar (shape [])
- **`mulScalar(a: Tensor, s: f32): Tensor`** — elementwise multiply by scalar
- **`divScalar(a: Tensor, s: f32): Tensor`** — elementwise divide by scalar
- **`slice(a: Tensor, start: i32, end: i32): Tensor`** — extract contiguous subvector
- **`concat(parts: Array<Tensor>): Tensor`** — concatenate vectors
- **`mul(a: Tensor, b: Tensor): Tensor`** — elementwise multiply, same shape (needed for attention dot product: `dot(q, k) = sum(mul(q, k))`)
- **`crossEntropy(logits: Tensor, targetId: i32): Tensor`** — fused softmax + negative log-likelihood; stores softmax probabilities in `cacheData` for backward
- **`scale(vec: Tensor, scalar: Tensor): Tensor`** — multiply vector by scalar tensor (needed for attention weighted sum where the weight must remain a differentiable Tensor)

### Leaf constructors

- **`tensorFrom(data: StaticArray<f32>, shape: StaticArray<i32>): Tensor`** — wrap existing data as a leaf (op = OP_NONE)
- **`tensorZeros(shape: StaticArray<i32>): Tensor`** — zero-initialized leaf
- **`tensorScalar(value: f32): Tensor`** — scalar tensor (shape [])

### Backward

**`backward(loss: Tensor): void`** — identical structure to scalar version:
1. DFS topological sort (iterative stack, not recursive, to avoid AS stack overflow on deep graphs)
2. Set `loss.grad[0] = 1.0`
3. Reverse walk: for each tensor, dispatch on `op` to accumulate gradients into children
4. Detach graph: null out `children` on all nodes for GC

The dispatch function `backwardOp(t: Tensor): void` is a top-level function with a switch/if-chain on `t.op`.

### Key differences from scalar autograd

| Scalar (consgpt.lisp) | Tensor |
|---|---|
| `Val` per f32 value | `Tensor` per matrix/vector |
| `localGrads: Array<f32>` per node | `op: i32` enum + dispatch function |
| `vadd(a, b)` creates 1 node for 1 scalar | `add(a, b)` creates 1 node for N elements |
| `linear(x, w)` = m*n vmul + m vsum nodes | `matmul(w, x)` = 1 node |
| Millions of nodes per forward pass | Dozens of nodes per forward pass |

## Step 2: `tests/tensor.test.ts` — Tensor autograd tests

### Forward pass tests
- `matmul`: known 2x3 matrix times 3-vector, verify result
- `add`: elementwise addition
- `relu`: positive passthrough, negative zeroed
- `softmax`: sums to 1.0, max-subtraction stability
- `rmsnorm`: output has unit RMS (up to epsilon)
- `logOp`: matches `Mathf.log` elementwise
- `neg`: sign flip
- `embedding`: correct row extracted
- `sum`: reduces to scalar
- `mulScalar`, `divScalar`: correct scaling
- `slice`: correct subvector
- `concat`: reassembles parts

### Backward pass tests
- `add` gradient: both inputs get output grad
- `matmul` gradient: verify against finite differences
- `relu` gradient: zero where input negative
- `softmax` gradient: verify against finite differences
- `rmsnorm` gradient: verify against finite differences
- `logOp` gradient: `grad / data`
- `embedding` gradient: only selected row accumulates

### Composed expression tests
- Chain of matmul + add + relu + matmul: gradients flow through
- Finite difference check on a small 2-layer network
- Graph detachment after backward (children are null)

## Step 3: `src/model.ts` — GPT-2 model

Port consgpt.lisp's `model.lisp` using tensor operations.

### Hyperparameters

Same as consgpt.lisp:
- `nEmbd: i32 = 64`
- `nLayer: i32 = 2`
- `nHead: i32 = 4`
- `headDim: i32 = 16` (nEmbd / nHead)
- `blockSize: i32 = 256`

### PRNG

Same LCG as consgpt.lisp:
- `lcgState: u32 = 42`
- `lcgNext(): u32` — `state = (1664525 * state + 1013904223) & 0xFFFFFFFF`
- `randomUniform(): f32` — `lcgNext() / 4294967296.0`
- `randomGaussian(): f32` — Box-Muller transform

### Weight initialization

**`makeMatrix(nout: i32, nin: i32): Tensor`** — allocate a [nout, nin] Tensor, fill with Gaussian * 0.02. Data is contiguous `StaticArray<f32>` of length `nout * nin`, row-major.

### State dictionary

`stateDict: Map<string, Tensor>` — each key maps to a single Tensor (not nested arrays).

**`params: Array<Tensor>`** — all weight tensors in sorted key order, for optimizer iteration.

**`initModel(vocabSize: i32): void`** — allocate all weight tensors:
- `wte`: [vocabSize, nEmbd] — token embeddings
- `wpe`: [blockSize, nEmbd] — position embeddings
- Per layer (i = 0..nLayer-1):
  - `layer{i}.attn_wq`: [nEmbd, nEmbd]
  - `layer{i}.attn_wk`: [nEmbd, nEmbd]
  - `layer{i}.attn_wv`: [nEmbd, nEmbd]
  - `layer{i}.attn_wo`: [nEmbd, nEmbd]
  - `layer{i}.mlp_fc1`: [4*nEmbd, nEmbd]
  - `layer{i}.mlp_fc2`: [nEmbd, 4*nEmbd]
- Weight tying: no separate `lm_head` — reuse `wte` for output projection

After allocation, collect all tensors into `params` in sorted key order.

Total: 14 tensors, same parameter count as consgpt.lisp.

### Forward pass

**`gpt(tokenId: i32, posId: i32, cacheKeys: Array<Array<Tensor>>, cacheVals: Array<Array<Tensor>>): Tensor`** — returns logits vector (length vocabSize).

```
x = add(embedding(wte, tokenId), embedding(wpe, posId))   // [nEmbd]
x = rmsnorm(x, 1e-5)

for li = 0 to nLayer-1:
  xResidual = x
  xn = rmsnorm(x, 1e-5)

  q = matmul(attn_wq, xn)    // [nEmbd]
  k = matmul(attn_wk, xn)    // [nEmbd]
  v = matmul(attn_wv, xn)    // [nEmbd]

  // KV cache: append k, v to per-layer cache
  cacheKeys[li].push(k)
  cacheVals[li].push(v)

  // Multi-head attention
  headOuts: Array<Tensor>     // collect per-head outputs
  for h = 0 to nHead-1:
    hs = h * headDim
    he = hs + headDim
    qH = slice(q, hs, he)                          // [headDim]
    // For each cached position t:
    //   score[t] = dot(qH, slice(cachedK[t], hs, he)) / sqrt(headDim)
    // attnWeights = softmax(scores)                // [seqLen]
    // headOut = sum over t: attnWeights[t] * slice(cachedV[t], hs, he)
    headOuts.push(headOut)                          // [headDim]

  xAttn = concat(headOuts)                          // [nEmbd]
  x = add(matmul(attn_wo, xAttn), xResidual)

  // MLP block
  xResidual = x
  x = rmsnorm(x, 1e-5)
  x = matmul(mlp_fc1, x)     // [4*nEmbd]
  x = relu(x)
  x = matmul(mlp_fc2, x)     // [nEmbd]
  x = add(x, xResidual)

// Output: weight tying with wte
logits = matmul(wte, x)       // [vocabSize]
```

### Attention detail: dot product and weighted sum

**Dot product:** `dot(qH, kH) = sum(mul(qH, kH))` — two graph nodes per dot product using `OP_MUL` (elementwise multiply) and `OP_SUM`.

**Attention scores:** Build a vector of scaled dot products across cached positions, then apply softmax:
```
scores = []
for t = 0 to seqLen-1:
  kH = slice(cacheKeys[li][t], hs, he)
  dotProd = sum(mul(qH, kH))                       // scalar tensor
  scaled = divScalar(dotProd, Mathf.sqrt(f32(headDim)))
  scores.push(scaled)
scoreVec = concat(scores)        // [seqLen]
attnWeights = softmax(scoreVec)  // [seqLen]
```

**Weighted sum:** Uses `OP_SCALE` to multiply each value vector by its corresponding attention weight. The attention weight must remain a differentiable Tensor (using `mulScalar` with a raw `f32` would break gradient flow through softmax):
```
headOut = scale(slice(cacheVals[li][0], hs, he), slice(attnWeights, 0, 1))
for t = 1 to seqLen-1:
  headOut = add(headOut, scale(
    slice(cacheVals[li][t], hs, he),
    slice(attnWeights, t, t+1)
  ))
```

All operations are differentiable tensor ops. Gradients flow through `scale` → `softmax` → `concat` → `divScalar` → `sum` → `mul` → `slice` back to q, k, v, and through the KV cache.

## Step 4: `src/train.ts` — Training entry point

Port consgpt.lisp's `train.lisp`. WASI CLI program.

### Data loading

- Call `initVocabulary()` to populate Pass 1 vocab
- Load merge rules from `build/merges.tsv` via `FileSystem.open()` + `parseMerges()`, then call `buildBpeVocab(merges, nextId)` to assign BPE IDs
- Read training corpus (WAT file) via `FileSystem.open()`, tokenize with `tokenize()`, encode each token via `vocab` lookup (Pass 1) or `bpeEncodeToken()` (Pass 2) into a flat `Array<i32>`

### Model initialization

- Call `initModel(vocabSize)` where vocabSize = `getBpeNextId()` (final ID after BPE vocab)

### Adam optimizer state

Per-tensor parallel arrays, stored as `Map<string, StaticArray<f32>>`:
- `adamM: Map<string, StaticArray<f32>>` — first moment, one entry per state dict key, same length as tensor's data
- `adamV: Map<string, StaticArray<f32>>` — second moment, same structure
- Both initialized to all zeros, same sizes as corresponding weight tensors

### Training loop

```
for i = 0 to numSteps-1:
  step = startStep + i
  batch = getBatch(step)       // contiguous window of trainSeqLen+1 IDs

  // Fresh KV cache per batch
  cacheKeys = new Array<Array<Tensor>>(nLayer)   // each starts empty
  cacheVals = new Array<Array<Tensor>>(nLayer)

  // Forward pass: autoregressive over sequence
  losses: Array<Tensor>
  for posId = 0 to trainSeqLen-1:
    tokenId = batch[posId]
    targetId = batch[posId + 1]
    logits = gpt(tokenId, posId, cacheKeys, cacheVals)
    probs = softmax(logits)
    // Cross-entropy loss for this position:
    // -log(probs[targetId])
    // Implemented as: neg(logOp(slice(probs, targetId, targetId+1)))
    // or index into probs.data[targetId] and wrap as scalar
    losses.push(loss_t)

  // Average loss
  loss = divScalar(sum(concat(losses)), f32(trainSeqLen))

  // Backward
  backward(loss)

  // Adam update
  lr_t = lr * (1.0 - f32(i) / f32(numSteps))     // linear decay

  for each tensor p in params (sorted key order):
    m = adamM.get(key)
    v = adamV.get(key)
    for j = 0 to p.data.length-1:
      g = p.grad[j]
      m[j] = beta1 * m[j] + (1 - beta1) * g
      v[j] = beta2 * v[j] + (1 - beta2) * g * g
      mHat = m[j] / (1 - beta1^(step+1))          // bias correction, absolute step
      vHat = v[j] / (1 - beta2^(step+1))
      p.data[j] -= lr_t * mHat / (sqrt(vHat) + eps)
      p.grad[j] = 0                                // zero grad

  // Print loss
  Console.error("step " + (step+1).toString() + " | loss " + loss.data[0].toString())

  // Checkpoint
  if ((i+1) % checkpointInterval == 0):
    saveCheckpoint(path, step+1, adamM, adamV)

// Final checkpoint if not aligned
if (numSteps % checkpointInterval != 0):
  saveCheckpoint(path, startStep + numSteps, adamM, adamV)
```

### Training configuration

Same as consgpt.lisp:
- `trainSeqLen: i32 = 32`
- `numSteps: i32 = 100`
- `lr: f32 = 0.001`
- `beta1: f32 = 0.9`
- `beta2: f32 = 0.999`
- `epsAdam: f32 = 1e-8`
- `checkpointInterval: i32 = 10`

### Batch extraction

**`getBatch(step: i32): StaticArray<i32>`** — extract a contiguous window of `trainSeqLen + 1` IDs from the corpus. Start index: `(step * trainSeqLen) % max(1, corpusLen - trainSeqLen - 1)`.

### Cross-entropy loss detail

For each position, we need `-log(probs[targetId])`. We use the fused `OP_CROSS_ENTROPY` (op 15): `crossEntropy(logits, targetId)`.

Forward: compute log-sum-exp for numerical stability, then `loss = -(logits[target] - max - log(sumExp))`. Store the softmax probabilities in `cacheData` for backward use (AS has no closures, so we cannot capture them).

Backward: `logits.grad[i] += upstream * (probs[i] - (i == targetId ? 1 : 0))`, where probs come from `cacheData`.

This is better than the composed version `neg(logOp(slice(softmax(logits), target, target+1)))` which creates 4 graph nodes per position (128 extra nodes for trainSeqLen=32) and is less numerically stable.

## Step 5: `src/checkpoint.ts` — Checkpoint persistence

### Checkpoint format: binary (`build/model.bin`)

Binary format — compact, natural for WASM, trivial to read/write via linear memory. The file is a flat sequence of little-endian values in a fixed order:

```
[i32]  step number
[i32]  nEmbd
[i32]  nLayer
[i32]  nHead
[i32]  blockSize
[i32]  vocabSize
[f32 x N]  weight values (all tensors in sorted key order, row-major)
[f32 x N]  Adam first moment (m) — one per parameter, same order
[f32 x N]  Adam second moment (v) — one per parameter, same order
```

The weight order follows consgpt.lisp's convention: state dict keys sorted alphabetically (`layer0.attn_wk`, `layer0.attn_wo`, ..., `layer1.mlp_fc2`, `wpe`, `wte`). Each tensor's data is already contiguous and row-major, so writing is a direct copy.

On load, the hyperparameters are validated against the current model config. Only the most recent checkpoint is needed — the file is overwritten on each save.

### Functions

- **`saveCheckpoint(path: string, step: i32, adamM: Map<string, StaticArray<f32>>, adamV: Map<string, StaticArray<f32>>): void`**
  - Write i32 header (step + 5 hyperparams)
  - For each key in sorted order: write tensor's `data` array
  - For each key in sorted order: write adamM entry
  - For each key in sorted order: write adamV entry

- **`loadCheckpoint(path: string): i32`** (returns step, or -1 if no checkpoint)
  - Read and validate header
  - For each key in sorted order: read into tensor's `data` array
  - For each key in sorted order: read into adamM entry
  - For each key in sorted order: read into adamV entry

### Binary I/O approach

Use raw WASI `fd_write`/`fd_read` with `changetype<usize>(staticArray)` to write/read `StaticArray<f32>` directly to/from the file descriptor. This avoids the overhead of `Descriptor.write()` which copies into a `u8[]` first. StaticArray data is stored inline starting at the object pointer, so `changetype<usize>(arr)` gives the address of the first element and `arr.length << 2` gives the byte count.

### Advantage over scalar version

With scalar autograd, checkpoint save extracts `val-data` from thousands of nested Val objects per matrix. With tensors, each weight's data is already a contiguous `StaticArray<f32>` — writing is essentially a memcpy.

## Step 6: Integration test

A smoke test that:
1. Tokenizes a small WAT snippet (e.g., `(module (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))`)
2. Initializes a model with the vocabulary size
3. Runs one forward pass through `gpt()`
4. Computes cross-entropy loss
5. Runs `backward(loss)`
6. Verifies weight gradients are non-zero (spot-check a few tensors)
7. Verifies graph is detached after backward

This validates the full pipeline without needing a full training run.

## Verification

1. `npm test` — all unit tests pass (tensor autograd, model, integration)
2. `npm run build` — compiles
3. Training smoke test: tokenize a small WAT file, run a few training steps, verify loss decreases

## Module dependency order

```
lexer.ts (done)
  └─ vocabulary.ts (done)
  └─ bpe.ts (done)
  └─ train-bpe.ts (done)
tensor.ts (independent)
model.ts (depends on tensor)
checkpoint.ts (depends on model)
train.ts (depends on all above)
```

## AssemblyScript constraints to watch for

- No closures — all functions top-level or class methods (this is why we use op enum dispatch instead of stored backward functions)
- No `for..of` — use index-based loops
- Field initializers required on classes
- Sort comparators must be top-level functions
- `<` after `.get()` is interpreted as generic type — store `.get()` result in local variable first
- `f32` arithmetic: may need explicit `f32()` casts to avoid implicit promotion
- `Map` keys must be value types or strings — no object keys
- `StaticArray<f32>` for contiguous tensor storage (not `Array<f32>` — StaticArray has fixed size and no indirection, maps cleanly to linear memory and future GPU buffers)
- `Mathf.log`, `Mathf.exp`, `Mathf.sqrt`, `Mathf.pow`, `Mathf.max`, `Mathf.abs` for f32 math (not `Math.log` which returns f64). All confirmed in AS standard library via `IMath<f32>` interface.
- Recursive DFS may hit stack limits on deep graphs — use iterative stack-based traversal for topological sort
- `Set<usize>` works for topological sort visited tracking — `usize` is a value type (`u32` in wasm32), and `changetype<usize>(tensor)` gives the unique object address
- `changetype<usize>(staticArray)` gives the memory address of a StaticArray's data — used for efficient binary I/O in checkpoint persistence

## Lessons learned (post-implementation)

### Implementation stats

- **tensor.ts**: ~530 lines (17 ops, Tensor class, forward ops, backward dispatch, topological sort)
- **model.ts**: ~180 lines (PRNG, weight init, state dict, full GPT-2 forward pass)
- **checkpoint.ts**: ~160 lines (raw WASI binary I/O, save/load with header validation)
- **train.ts**: ~210 lines (CLI parsing, data loading, Adam optimizer, training loop)
- **tests**: 600 total (tensor: 130, model: 42, integration: 5, plus existing lexer/vocab/BPE tests)
- **Total parameter count**: 121,088 (with vocab=100), actual training uses vocab=757

### Training smoke test results

- Vocabulary: 568 pass-1 tokens + 189 BPE merges = 757 total
- Corpus: 53,102 tokens → 96,759 encoded token IDs
- Loss decrease over 5 steps: 6.58 → 6.12 (healthy gradient descent)
- Initial loss ~6.58 ≈ ln(757) ≈ 6.63 (near-random prediction, as expected)

### Issues encountered during implementation

1. **`concat` backward bug**: Initially wrote `child.grad[offset + j]` — wrong because each child's grad array is indexed from 0 within that child. The correct code is `child.grad[j] += g[offset + j]` where `offset` indexes into the output grad, not the child grad.

2. **`rmsnorm` backward complexity**: The backward rule for RMS normalization has a non-trivial chain rule. The clean formula is `a.grad[i] += rmsScale * (g[i] - t.data[i] * dotGOut / n)` where `rmsScale = 1/rms`, `dotGOut = sum(g * t.data)`, and `n` is the vector length.

3. **`tensorSum` naming**: Named `tensorSum` instead of `sum` to avoid potential naming conflicts in AssemblyScript.

4. **Graph detachment scope**: `backward()` detaches children (sets to empty array) but does not reset `op`. This is correct — op is metadata, children are what hold the graph alive for GC.

5. **`crossEntropy` cacheData pattern**: Storing softmax probabilities in `cacheData` during forward for use in backward works well as a closure substitute. This pattern could be reused for any op that needs intermediate results during backward.

### Deviations from plan

- **Loss computation**: The plan's Step 4 pseudocode showed `softmax` + `neg(logOp(slice(...)))` for loss, but the actual implementation uses the fused `crossEntropy` op (as specified in the cross-entropy detail section). No deviation, just the plan had two descriptions.
- **`loadCheckpoint` signature**: Takes `adamM` and `adamV` as parameters (not just path), since it needs to restore optimizer state. The plan's Step 5 listed the simpler signature but the implementation correctly passes optimizer state.

### What went well

- The op enum dispatch pattern is clean and extensible — adding a new op requires: (1) constant, (2) forward function, (3) backward case in `backwardOp`.
- Finite-difference gradient checking caught no bugs beyond the concat/rmsnorm issues found during initial development.
- The binary checkpoint format is simple and efficient — just raw memory dumps in sorted key order.
- Integration tests validate the full pipeline in ~100 lines of test code.
