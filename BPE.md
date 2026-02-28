# Plan: BPE Training Pipeline

## Context

The tokenizer is complete: the lexer splits WAT into token strings, the vocabulary assigns fixed IDs to known tokens, and the BPE module can train and encode — but it has no trained merge rules yet. Every token not in the fixed vocabulary (numeric literals, identifiers, strings) currently has no encoding.

This plan builds the pipeline to generate a WAT corpus from our own source code, train BPE merge rules on it, and persist them for use during model training.

Comments are stripped by the lexer (`src/lexer.ts`), so watnot's injected comments won't affect tokenization yet, but running the full annotation pipeline exercises all our tooling and produces annotated WAT for future use.

## Step 0: Update BPE.md with research findings

Update this file with any findings from pre-implementation research.

## Step 1: Add corpus generation scripts to package.json

Add scripts that chain the existing tools to produce annotated WAT from wasmgpt's own source:

- **`wat`** — run `wasm2wat` on `build/wasmgpt.wasm` to produce `build/wasmgpt.wat` and `build/wasmgpt.offsets.json`
- **`annotate`** — run watnot on the WAT + source map + offset map + source files to produce `build/wasmgpt.annotated.wat`
- **`corpus`** — chain `build`, `wat`, and `annotate` to produce the full corpus from scratch

Tool paths (assumed on PATH):
- `wasm2wat` — from [our fork of wabt](https://github.com/spratt/wabt/tree/byte_offsets) with `--offset-map` support
- `wasmtime` — WASI runtime for running watnot and train-bpe
- watnot: git submodule at `watnot/`, built via `npm run setup:watnot`

watnot CLI arg order:
```
<source.wasm.map> <source.wat> <source.offsets.json> <source1.ts> [source2.ts] ...
```

wasmgpt source files: `src/index.ts src/lexer.ts src/vocabulary.ts src/bpe.ts`

The `wat` and `annotate` scripts depend on `build` having been run first. The `corpus` script chains all three.

The annotated WAT goes to stdout from watnot, so `annotate` should redirect to `build/wasmgpt.annotated.wat`.

## Step 2: `src/train-bpe.ts` — BPE training entry point

A WASI CLI program following the same pattern as watnot: reads files via `FileSystem.open()` with CLI args, writes output to stdout via `Console.write()`.

Usage:
```
wasmtime --dir . build/train-bpe.wasm <corpus.wat> [numMerges]
```

1. Reads WAT text from the file path in `args[1]` (using `FileSystem.open()` + `readString()`, same as watnot's `readFileText()`)
2. Reads optional `numMerges` from `args[2]` (default: 256)
3. Calls `tokenize()` to get token strings
4. Calls `initVocabulary()` to populate the fixed vocab
5. Partitions tokens: those in `vocab` are Pass 1, the rest are unknowns
6. Calls `trainBpe(unknowns, numMerges)` to learn merge rules
7. Writes merge rules to stdout in a simple format (one merge per line, tab-separated: `left\tright`), using watnot's `writeOutput()` pattern to avoid the `as-wasi` `writeStringLn` bug

Add package.json scripts:
- **`build:train-bpe`** — compile `src/train-bpe.ts` to `build/train-bpe.wasm`
- **`train:bpe`** — run the full pipeline: generate corpus, then run the BPE trainer on `build/wasmgpt.annotated.wat`, redirect stdout to `build/merges.tsv`

## Step 3: Add merge persistence to `src/bpe.ts`

Add functions to parse and serialize the merge format:

- **`parseMerges(text: string): Array<Array<string>>`** — parse tab-separated merge rules
- **`serializeMerges(merges: Array<Array<string>>): string`** — serialize to tab-separated format

These are used by `train-bpe.ts` (serialize) and eventually by `train.ts` (parse).

## Step 4: `tests/train-bpe.test.ts` — Tests

- `parseMerges` / `serializeMerges` round-trip
- Token partitioning: known tokens go to Pass 1, unknown tokens go to BPE training
- Empty corpus produces no merges
- Merge output format is correct (tab-separated, one per line)

## Step 5: Verification

1. `npm test` — all tests pass
2. `npm run build` — compiles
3. `npm run corpus` — produces `build/wasmgpt.annotated.wat`
4. `npm run train:bpe` — produces `build/merges.tsv` with learned merge rules
5. Inspect `build/merges.tsv` to verify merge rules look reasonable

## Step 6: Update BPE.md with lessons learned

Update this file with:
- Actual number of unknown tokens in the corpus
- Number of merges learned
- Any issues encountered with WASI I/O, wasmtime, or the pipeline

## Module dependency order

```
lexer.ts (done)
  └─ vocabulary.ts (done)
  └─ bpe.ts (done, adding parseMerges/serializeMerges)
train-bpe.ts (new, depends on lexer + vocabulary + bpe)
```

## AssemblyScript constraints to watch for

- WASI file reading: use `FileSystem.open(path, "r")` + `(fd as Descriptor).readString()`, same as watnot
- WASI stdout: use `Console.write(s, false)` — the `as-wasi` `writeStringLn()` bug overwrites the newline byte (see watnot daily log for details)
- CLI args via `CommandLine.all` — `args[0]` is the program name
- `I32.parseInt()` for parsing the optional numMerges CLI arg

## Post-implementation lessons learned

### Corpus statistics
- wasmgpt's own compiled WAT: 11,849 lines (annotated), 11,784 lines (raw)
- 53,102 total tokens after lexing
- 42,187 known tokens (in Pass 1 vocabulary, 79.4%)
- 10,915 unknown tokens (need BPE, 20.6%)
- 256 merge rules learned (default numMerges)

### Tree-shaking
- The initial `src/index.ts` was a placeholder with an unused import, so `npm run build` produced a 12-line WAT with all modules tree-shaken out. Added exported functions that exercise lexer, vocabulary, and BPE to force the modules into the compiled output.

### Hardcoded paths
- Initial scripts used hardcoded absolute paths (`$HOME/Personal/wabt/build/wasm2wat`, `$HOME/.wasmtime/bin/wasmtime`, `$HOME/Personal/watnot/build/watnot.wasm`).
- Fixed by: adding watnot as a git submodule, creating symlinks in `~/bin` for `wasm2wat` and `wasmtime`, and assuming both are on PATH.
- Added `setup:watnot` script to install and build the submodule.

### watnot submodule
- `watnot.wasm` is not tracked in watnot's git repo (it's in `.gitignore`), so users must run `npm run setup:watnot` after cloning to build it.

### Diagnostic output
- `train-bpe.ts` writes statistics (token counts, merge count) to stderr so they don't contaminate the TSV output on stdout.
