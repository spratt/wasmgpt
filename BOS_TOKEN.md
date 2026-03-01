# Plan: Add BOS Token

## Context

wasmgpt has no BOS (beginning-of-sequence) token. Inference starts from `(` (ID 0) and runs until `BLOCK_SIZE` (256 positions). There is no way for the model to signal that generation is complete, so every sample is exactly 256 tokens long regardless of whether the output is a complete program or trailing garbage.

consgpt.lisp uses a BOS token that serves double duty: inference starts from BOS, and if the model generates BOS, generation stops. This lets the model learn natural program boundaries.

## How microgpt.py does it

Karpathy's microgpt wraps each training document with BOS on both sides:

```python
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

This teaches the model two things:
- BOS → first token of a program (what to generate after start)
- last token → BOS (when a program is complete)

Inference starts from BOS and stops when BOS is generated:

```python
token_id = BOS
```

consgpt.lisp registers BOS but does NOT inject it into training data — the model never learns when to generate it. We follow microgpt's approach instead.

## Plan for wasmgpt

### Step 1: Register BOS in `vocabulary.ts`

BOS is a static control token — it never appears in the corpus, never participates in BPE character splitting or pair merging. It belongs in Pass 1 alongside `(`, `)`, `module`, etc.

Add `reg("<BOS>")` at the end of `initVocabulary()`:

```typescript
export function initVocabulary(): void {
  // ... existing registrations ...

  // --- Control tokens ---
  reg("<BOS>");
}
```

BOS gets a fixed Pass 1 ID (568, since the current Pass 1 has IDs 0–567). This means:
- `getNextId()` returns 569 instead of 568
- `buildBpeVocab(merges, 569)` starts BPE IDs from 569
- All BPE IDs shift up by 1
- `getBpeNextId()` returns 877 instead of 876

No changes needed in `train-bpe.ts` or `train.ts` for registration — both already call `initVocabulary()`, which now includes BOS. `serializeVocab()` already includes all Pass 1 vocab entries, so `vocab.sexp` will automatically contain `<BOS>`.

Existing checkpoints become incompatible (vocab size mismatch in checkpoint header).

### Step 2: Export BOS ID from `vocabulary.ts`

Add a getter so other modules can find the BOS ID without hardcoding it:

```typescript
export function getBosId(): i32 {
  return vocab.get("<BOS>");
}
```

### Step 3: Inject BOS into training data (`train.ts`)

Following microgpt's approach, wrap the encoded corpus with BOS:

```typescript
const bosId = getBosId();
allIds.unshift(bosId);  // prepend BOS
allIds.push(bosId);     // append BOS
```

This ensures the model sees BOS → first-token at the start and last-token → BOS at the end. Training windows that cross the boundary will include BOS, teaching the model to generate it at sequence boundaries.

### Step 4: Update `infer.ts`

Inference already computes vocab size as `max(ID) + 1` from `vocab.sexp`, so it will automatically pick up the new vocab size (877).

Changes needed:

1. **Find BOS ID:** Look up `<BOS>` in `tokenToId` after loading vocab:
   ```typescript
   const BOS_TOKEN = "<BOS>";
   if (!tokenToId.has(BOS_TOKEN)) {
     Console.error("ERROR: no <BOS> token in vocabulary\n");
     abort();
   }
   const BOS_ID: i32 = tokenToId.get(BOS_TOKEN);
   ```

2. **Start from BOS:** Change the initial `tokenId` from `0` (which is `(`) to `BOS_ID`:
   ```typescript
   let tokenId: i32 = BOS_ID;  // was: 0
   ```

3. **Stop on BOS:** In the generation loop, break if the model generates BOS:
   ```typescript
   if (nextId == BOS_ID) break;
   ```

4. **Prompt handling:** When a prompt is provided, feed BOS at position 0, then feed prompt tokens. The prompt tokens build the KV cache before autoregressive generation begins.

### Step 5: Retrain

Adding BOS changes the vocab size (876 → 877), which changes the `wte` matrix dimensions. Full retrain required:

```bash
npm run clean:model
npm run train:bpe        # regenerates merges.sexp and vocab.sexp with <BOS>
npm run train -- 1000    # train from scratch
npm run infer            # verify BOS start/stop behavior
```

### Step 6: Tests

- Verify `<BOS>` is in `vocab` after `initVocabulary()`
- Verify `vocab.get("<BOS>")` returns 568
- Verify `getNextId()` returns 569 after init
- Verify inference stops when BOS is generated (mock test)

## Files to modify

| File | Change |
|------|--------|
| `src/vocabulary.ts` | Add `reg("<BOS>")` at end of `initVocabulary()`, add `getBosId()` |
| `src/train.ts` | Import `getBosId`, wrap `allIds` with BOS |
| `src/infer.ts` | Look up BOS ID, start from BOS, stop on BOS |

## Impact

- Pass 1 vocab: 568 → 569
- Total vocab: 876 → 877
- All BPE IDs shift up by 1
- Checkpoint: incompatible, retrain required
- Model params: +64 (one new row in `wte`, negligible)
- Inference behavior: variable-length output (stops at BOS or BLOCK_SIZE)

## Resolved: BOS in training data

microgpt.py wraps training data with BOS; consgpt.lisp does not. We follow microgpt's approach because without BOS in training data, the model has no signal to learn when to stop generating.
