# WasmGPT

[![CI](https://github.com/spratt/wasmgpt/actions/workflows/ci.yml/badge.svg)](https://github.com/spratt/wasmgpt/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)

A GPT-style language model trained on WebAssembly. Given a WAT prefix, the model generates a valid and plausible continuation. The project is implemented in AssemblyScript so that its own source code — compiled to annotated WAT via [watnot](https://github.com/spratt/watnot) — can be included in the training corpus.

Based on [consgpt.lisp](https://github.com/spratt/consgpt.lisp), a GPT trained on Common Lisp. WAT and Common Lisp are both S-expression languages, so the hybrid tokenizer architecture, model design, and training pipeline carry over with targeted changes.

## Prerequisites

- [Node.js](https://nodejs.org/) (v22+)

## Setup

```
npm install
```

## Build

```
npm run build
```

## Test

```
npm test
```

## Design

See [DESIGN.md](DESIGN.md) for architecture, tokenization strategy, corpus generation, and design decisions.
