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

- **Hybrid tokenizer architecture** — consgpt.lisp uses a two-pass tokenizer: Pass 1 assigns fixed integer IDs to all 978 ANSI CL symbols plus syntax tokens, Pass 2 uses BPE for user-defined names, numbers, and strings. WasmGPT replaces the 978 CL symbols with ~436 WASM instruction opcodes but keeps the same two-pass design. Both projects face the same split: a closed, enumerable set of language primitives plus open-ended user content.

- **BPE training and encoding** — the BPE algorithm (character splitting, pair counting, iterative merging, subword encoding) is language-agnostic. The merge training (`train-bpe.lisp`) and encoding (`bpe.lisp`) logic ports directly.

- **GPT-2 model architecture** — consgpt.lisp implements a GPT-2 style transformer with RMSNorm, weight tying (`lm_head` shares weights with `wte`), multi-head self-attention with KV cache, and a ReLU MLP block. The architecture is identical for WasmGPT; only the vocabulary size changes.

- **Scalar autograd** — differentiable operations (`vadd`, `vmul`, `vpow`, `vlog`, `vexp`, `vrelu`) with DFS topological sort backward pass. The autograd engine is domain-independent and ports directly.

- **Adam optimizer** — with linear LR decay, bias correction, and per-parameter first/second moment tracking.

- **Training loop** — batch extraction, forward pass, loss computation (cross-entropy via negative log probability), backward pass, Adam update, checkpointing.

- **Inference loop** — autoregressive generation with temperature-scaled sampling, KV cache with graph detachment for memory management.

### What changes

| Component | consgpt.lisp | WasmGPT |
|---|---|---|
| Implementation language | Common Lisp (SBCL) | AssemblyScript (compiled to WASM) |
| Fixed vocabulary | 978 ANSI CL symbols + 10 syntax tokens + 15 LOOP keywords + 13 common keywords = 1,016 | ~436 WASM 3.0 instruction opcodes + WAT syntax tokens (`(`, `)`, `;;`, etc.) |
| Lexer | Common Lisp reader (atoms, strings, dispatch macros, `#\|...\|#` block comments) | WAT S-expression parser (mnemonics, immediates, identifiers, `;;` comments) |
| Numeric precision | `double-float` (64-bit) | `f32` (32-bit) — matches WASM native type and WebGPU shader constraints |
| Data structures | Hash tables, cons lists, defstruct | Maps, Arrays, classes with field initializers |
| Runtime | SBCL native compilation, 16GB heap | WASM + WebGPU in the browser |
| Checkpointing | S-expressions written with `*print-readably*` | TBD — likely ArrayBuffer or IndexedDB |
| Memory management | SBCL GC with graph detachment after backward | Manual or GC depending on AS runtime; same detachment pattern |

### Lexer adaptation

The consgpt.lisp lexer (`lexer.lisp`) handles CL-specific syntax: `#'` (function), `#\` (character), `#(` (vector), `` ` `` (backquote), `,` and `,@` (unquote). The WAT lexer is simpler — WAT has only parentheses, atoms (instruction mnemonics, `$identifiers`, numeric literals), and `;;` comments. No dispatch macros, no reader macros, no string escapes beyond what WAT defines. The lexer is smaller but must recognize WAT instruction mnemonics to assign them fixed token IDs in Pass 1.

### Model size

consgpt.lisp's Tiny config (E=64, L=2, H=4, B=256, V=1,382) has 203K parameters and trains in ~10 minutes per 100 steps on CPU. WasmGPT's vocabulary will be smaller (~500-600 total tokens vs. 1,382), further reducing embedding and output matrix dimensions. The same Tiny config is a reasonable starting point, with the option to scale up once the pipeline is validated.

---

## Architecture

The model follows the GPT-2 architecture as implemented in [consgpt.lisp](https://github.com/spratt/consgpt.lisp):

- Transformer with multi-head self-attention and KV cache
- RMSNorm (not LayerNorm) — simpler and sufficient at small scale
- Weight tying between token embedding and output projection
- ReLU activation in the MLP block
- Scalar autograd engine (no framework dependency)
- Adam optimizer with linear LR decay and bias correction
- Cross-entropy loss
- Training loop with checkpointing, inference loop with temperature sampling

Training targets browser deployment from the outset. The inference path uses WebGPU for matrix operations and WASM for CPU-side orchestration (autograd, Adam, tokenizer, training loop).

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

1. **Pass 1: Instruction tokens** — each WAT instruction opcode gets a dedicated token ID, analogous to how consgpt.lisp assigns fixed IDs to all 978 ANSI CL symbols. With ~436 instructions in WASM 3.0 (including SIMD), this is a small, fixed vocabulary. Binary opcodes and WAT mnemonics share the same token ID since they are two representations of the same instruction.

2. **Pass 2: BPE tokens** — everything else (immediates, identifiers, comments, data) is tokenized via Byte Pair Encoding trained on the corpus. This is the same BPE algorithm used in consgpt.lisp: character splitting, weighted pair counting, iterative merging, and subword encoding with an `<UNK>` fallback.

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
| Tokenizer | Two-pass hybrid (instruction tokens + BPE) | Same design as consgpt.lisp; exploits known WASM structure; BPE handles open-ended values |
| Corpus annotation | Source-map-based comment transplantation via [watnot](https://github.com/spratt/watnot) | Human-authored intent without synthetic generation |
| Deployment | Browser (WASM + WebGPU) | Aligns with domain; no server infrastructure required |
| Training trigger | WAT code completion | Keeps model simple; natural language deferred to fine-tuning phase |
| Self-reference | Project source in corpus | Elegant bootstrapping; increases corpus coverage of idiomatic WAT |

---

## Open Questions

- What vocabulary size is appropriate for the BPE component given the expected corpus size?
- Should the model be trained on WAT, binary, or a mix? (Binary gives more data volume; WAT gives better annotation quality)
- How should the tokenizer handle WASM 3.0 extensions vs. MVP instructions — unified vocabulary or versioned?
- What is a reasonable model size given browser memory constraints for inference?
- How to handle the structured block nesting in WAT (blocks must be well-formed) — should this be enforced at the tokenizer level or left to the model to learn?
