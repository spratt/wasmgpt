# Plan: Tokenizer (Vocabulary + BPE)

## Context

The WAT lexer (`src/lexer.ts`) splits WAT text into raw token strings. The tokenizer builds on the lexer to assign integer IDs to every token, using a two-pass hybrid design ported from consgpt.lisp.

## Pre-implementation research findings

- wabt's `opcode.def` yields **543 unique instruction mnemonics** (excluding interpreter-only opcodes marked `Interp`). The only duplicate text mnemonic is `select` (appears for both `Select` and `SelectT` enum entries).
- **Structural WAT keywords** not already in the opcode list: `module`, `type`, `func`, `import`, `export`, `memory`, `table`, `global`, `data`, `elem`, `start`, `param`, `result`, `local`, `mut`, `offset`, `align`
- **Type names** not in opcode list: `i32`, `i64`, `f32`, `f64`, `v128`, `funcref`, `externref`
- **Syntax tokens**: `(`, `)`
- Some keywords (`block`, `loop`, `if`, `else`, `end`, `br`, `call`, `drop`, `select`, `return`, `nop`, `unreachable`) overlap with instruction mnemonics — `reg()` is idempotent so no conflict.
- **Estimated total fixed vocabulary**: ~569 tokens after deduplication.
- **BPE merge representation**: `Array<Array<string>>` where each inner array is `[left, right]`. Simplest AS-compatible representation of consgpt.lisp's cons cell `(left . right)` pairs.
- **Shared state**: `nextId` counter is shared between vocabulary and BPE so BPE IDs continue where Pass 1 left off.

## Step 1: `src/vocabulary.ts` — Pass 1 instruction vocabulary

Register all WASM instruction mnemonics and WAT syntax tokens with fixed integer IDs. Analogous to consgpt.lisp's `symbol-tokenizer.lisp`.

- **`vocab: Map<string, i32>`** — maps token string to fixed ID
- **`nextId: i32`** — tracks the next available ID
- **`reg(name: string): void`** — register a single token
- **`regAll(names: Array<string>): void`** — register a list of tokens
- **`initVocabulary(): void`** — registers all tokens in order:
  1. WAT syntax tokens: `(`, `)`
  2. WAT structural keywords: `module`, `func`, `param`, `result`, `local`, `global`, `import`, `export`, `memory`, `table`, `type`, `elem`, `data`, `offset`, `mut`, `start`, `block`, `loop`, `if`, `then`, `else`, `end`, `br`, `br_if`, `br_table`, `return`, `call`, `call_indirect`, `drop`, `select`, `unreachable`, `nop`
  3. Type names: `i32`, `i64`, `f32`, `f64`, `v128`, `funcref`, `externref`
  4. All remaining WASM instruction mnemonics from wabt's `opcode.def` (~500 more), grouped by category (memory, numeric, SIMD, atomic, etc.)

Source for mnemonics: extract from `/home/spratt/Personal/wabt/include/wabt/opcode.def` using:
```bash
grep '^WABT_OPCODE' opcode.def | grep -v 'Interp' | awk -F'"' '{print $2}' | sort -u
```

Exclude interpreter-only opcodes (marked `Interp` in the Name field). Deduplicate entries that share the same text mnemonic (e.g., `select` appears twice).

## Step 2: `tests/vocabulary.test.ts` — Vocabulary tests

- All registered tokens have unique IDs
- Known mnemonics are present (`i32.add`, `local.get`, `call`, `nop`, etc.)
- Syntax tokens `(` and `)` are present
- Structural keywords are present
- No duplicate IDs exist (iterate vocab, collect IDs, check size equals count)
- `nextId` equals total vocabulary size after initialization

## Step 3: `src/bpe.ts` — Pass 2 BPE

Port consgpt.lisp's `bpe.lisp`. The BPE algorithm is language-agnostic.

- **`tokenToChars(token: string): Array<string>`** — split token into single characters
- **`mergePair(pair: string[], word: Array<string>): Array<string>`** — apply one merge
- **`countPairs(wordFreqs: Map<string, i32>): Map<string, i32>`** — count adjacent pairs weighted by frequency. Key encoding: `left + "\0" + right` (since AS Maps need string keys)
- **`trainBpe(unknownTokens: Array<string>, numMerges: i32): Array<string[]>`** — learn merge rules
- **`applyMerges(chars: Array<string>, merges: Array<string[]>): Array<string>`** — apply all merges to a character list
- **`bpeEncodeToken(token: string): Array<i32>`** — encode a token to BPE IDs
- **`buildBpeVocab(merges: Array<string[]>): void`** — assign IDs starting from `nextId`

Persistence (save/load merge rules) can use a simple newline-delimited format since we don't have S-expressions. Deferred until we have file I/O needs.

## Step 4: `tests/bpe.test.ts` — BPE tests

Port consgpt.lisp's `bpe-test.lisp`:
- Character splitting
- Single merge application
- Pair counting with frequencies
- BPE training on a small corpus
- Encoding with learned merges
- Unknown token fallback (`<UNK>`)
- Vocabulary building (IDs start after Pass 1)

## Verification

1. `npm test` — all vocabulary and BPE tests pass
2. `npm run build` — compiles

## Module dependency order

```
lexer.ts (done)
  └─ vocabulary.ts (Pass 1)
  └─ bpe.ts (Pass 2, IDs start after vocabulary)
```

## AssemblyScript constraints to watch for

- No closures — all functions top-level or class methods
- No `for..of` — use index-based loops
- `<` after `.get()` is interpreted as generic type — store `.get()` result in local variable first
- `Map` keys must be value types or strings — no object keys

## Post-implementation lessons learned

### Final vocabulary size
- 568 string literals registered in `vocabulary.ts` (including overlapping keywords/instructions)
- After deduplication via idempotent `reg()`, the actual vocab size depends on how many structural keywords overlap with instruction mnemonics (e.g., `block`, `loop`, `if`, `else`, `end`, `br`, `br_if`, `br_table`, `return`, `call`, `call_indirect`, `drop`, `select`, `unreachable`, `nop` — 15 overlaps)
- Estimated unique tokens: ~553

### BPE implementation decisions
- **Word encoding**: Words (arrays of subword strings) are encoded as `\x01`-separated strings for use as Map keys, since AS Maps require string keys. The `\0` separator is used for pair keys (left + `\0` + right).
- **Shared `nextId`**: Rather than re-exporting a mutable binding (which AS doesn't support cleanly), `buildBpeVocab` takes `startId` as a parameter. The caller passes `nextId` from vocabulary.ts.
- **Sort comparator**: AS requires top-level sort comparators — `stringLessThan` is a top-level function returning `i32`.
- **`regAll` uses `StaticArray<string>`**: More efficient than `Array<string>` for hardcoded constant arrays in the vocabulary.

### Test framework limitations
- `assemblyscript-unittest-framework` does not have `toBeGreaterThan` — use `expect(x > y).equal(true)` instead.

### Coverage
- vocabulary.ts: 100% statements, 100% lines, 100% functions, 75% branches (the only missed branch is the `vocab.has()` check in `reg()` which is tested via the idempotency test)
- bpe.ts: 99.14% statements, 92.65% branches, 100% functions, 99.14% lines
- All 224 test cases pass (lexer + vocabulary + BPE)
