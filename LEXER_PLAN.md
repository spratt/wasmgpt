# Plan: WasmGPT Project Scaffolding and WAT Lexer

## Context

WasmGPT is a GPT trained on WebAssembly, based on consgpt.lisp. The first implementation step is project scaffolding (mirroring the watnot sister project) and a WAT lexer that tokenizes WAT text into a flat list of token strings. The lexer is the foundation for the hybrid tokenizer — Pass 1 (instruction vocabulary) and Pass 2 (BPE) both consume its output.

## Step 1: Project scaffolding

Create these files, mirroring watnot's patterns:

- **`package.json`** — name "wasmgpt", same devDependencies as watnot (assemblyscript ^0.27, @assemblyscript/wasi-shim ^0.1, as-wasi ^0.6.0, assemblyscript-unittest-framework ^2.1.0), build/test scripts
- **`asconfig.json`** — extends wasi-shim config
- **`as-test.config.js`** — include src + tests, exclude src/index.ts
- **`.gitignore`** — node_modules/, build/
- **`src/index.ts`** — minimal placeholder so `npm run build` works

Then run `npm install`.

## Step 2: `src/lexer.ts` — WAT lexer

A single exported function `tokenize(text: string): Array<string>` that splits WAT text into tokens. Comments are stripped. Tokens are returned verbatim (case-sensitive, no normalization).

**Token types handled:**
- `(` and `)` — single-character tokens
- `;;` line comments — skipped (advance to newline)
- `(; ... ;)` block comments — skipped with nesting support (depth counter)
- `"..."` string literals — includes quotes, handles `\` escapes
- Atoms — everything else until whitespace, `(`, `)`, `"`, or `;`

**Helper functions** (all top-level, no closures per AS constraint):
- `isWhitespace(ch: i32): bool` — space, tab, newline, CR
- `isTerminator(ch: i32): bool` — whitespace + `(`, `)`, `"`, `;`
- `skipWhitespace(text, pos, len): i32`
- `skipLineComment(text, pos, len): i32`
- `skipBlockComment(text, pos, len): i32` — tracks depth for nested `(; ;)`
- `readStringLiteral(text, pos, len): i32` — handles `\` escapes
- `readAtom(text, pos, len): i32` — reads until terminator

**Design decisions:**
- Atoms are NOT upcased (WAT is case-sensitive, unlike CL)
- Dispatch order: line comment → block comment → `(` → `)` → `"` → atom
- `(;` checked before `(` to avoid emitting `(` for comment starts
- `;` in terminator set so atoms stop before `;;` comments even without whitespace

## Step 3: `tests/lexer.test.ts` — Unit tests

Following watnot's test patterns: import from `assemblyscript-unittest-framework/assembly`, use `test()` + `expect().equal()`.

Test categories:
- Empty/whitespace input
- Parentheses (single, nested)
- Atoms: instruction mnemonics (`nop`, `i32.add`), identifiers (`$func_name`, `$~lib/rt/itcms/state`), numeric literals (`42`, `0xFF`, `-1`, `0.5`), keywords (`module`, `func`, `param`)
- String literals: simple, with escapes, with escaped quotes, empty
- Line comments: standalone, after token, comment-only input
- Block comments: simple, nested, between tokens
- Integration: real WAT snippets (module definition, function with comments)

## Verification

1. `cd /home/spratt/Personal/wasmgpt && npm install`
2. `npm test` — all lexer tests pass
3. `npm run build` — compiles to `build/wasmgpt.wasm`

## Reference files
- `/home/spratt/Personal/watnot/package.json` — scaffolding pattern
- `/home/spratt/Personal/watnot/src/comments.ts` — AS coding style (charCodeAt, manual parsing)
- `/home/spratt/Personal/consgpt.lisp/lexer.lisp` — algorithm being adapted
