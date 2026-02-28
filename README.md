# WasmGPT

[![CI](https://github.com/spratt/wasmgpt/actions/workflows/ci.yml/badge.svg)](https://github.com/spratt/wasmgpt/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)

A GPT-style language model trained on WebAssembly. Given a WAT prefix, the model generates a valid and plausible continuation. The project is implemented in AssemblyScript so that its own source code — compiled to annotated WAT via [watnot](https://github.com/spratt/watnot) — can be included in the training corpus.

Based on [consgpt.lisp](https://github.com/spratt/consgpt.lisp), a GPT trained on Common Lisp. WAT and Common Lisp are both S-expression languages, so the hybrid tokenizer architecture, model design, and training pipeline carry over with targeted changes.

## Prerequisites

- [Node.js](https://nodejs.org/) (v22+)
- [wasmtime](https://wasmtime.dev/) — for running WASI programs
- [wasm2wat](https://github.com/spratt/wabt/tree/byte_offsets) — from our fork of wabt with `--offset-map` support (build from source, ensure `wasm2wat` is on PATH)

## Setup

Clone with submodules:

```
git clone --recurse-submodules https://github.com/spratt/wasmgpt.git
cd wasmgpt
```

If you already cloned without `--recurse-submodules`:

```
git submodule update --init --recursive
```

Install dependencies and build the watnot submodule:

```
npm install
npm run setup:watnot
```

## Build

```
npm run build
```

## Test

```
npm test
```

## Scripts

| Script | Description |
|---|---|
| `build` | Compile `src/index.ts` to `build/wasmgpt.wasm` (debug + source map) |
| `build:release` | Compile with optimizations |
| `build:train-bpe` | Compile `src/train-bpe.ts` to `build/train-bpe.wasm` |
| `setup:watnot` | Install and build the watnot submodule |
| `wat` | Disassemble `build/wasmgpt.wasm` to WAT with offset map |
| `annotate` | Inject source comments into the WAT using watnot |
| `corpus` | Full pipeline: build + wat + annotate |
| `train:bpe` | Full pipeline: corpus + build train-bpe + train BPE merge rules |
| `test` | Run unit tests |

## Design

See [DESIGN.md](DESIGN.md) for architecture, tokenization strategy, corpus generation, and design decisions.
