// train-bpe: Train BPE merge rules on a WAT corpus.
// Reads a WAT file, tokenizes it, partitions tokens into known (Pass 1)
// and unknown, trains BPE on the unknowns, and writes merge rules as TSV.
//
// Usage:
//   wasmtime --dir . build/train-bpe.wasm <corpus.wat> [numMerges]

import { CommandLine, Console, FileSystem, Descriptor } from "as-wasi/assembly";
import { tokenize } from "./lexer";
import { vocab, nextId, initVocabulary } from "./vocabulary";
import { trainBpe, serializeMerges } from "./bpe";

// --- Parse CLI args ---

const args = CommandLine.all;

if (args.length < 2) {
  Console.error("Usage: train-bpe <corpus.wat> [numMerges]\n");
  abort();
}

const corpusPath = args[1];
let numMerges: i32 = 256;
if (args.length >= 3) {
  numMerges = I32.parseInt(args[2]);
}

// --- Read corpus ---

const fd = FileSystem.open(corpusPath, "r");
if (fd === null) {
  Console.error("Error: could not open file: " + corpusPath + "\n");
  abort();
}
const watText = (fd as Descriptor).readString();
if (watText === null) {
  Console.error("Error: could not read file: " + corpusPath + "\n");
  abort();
}

// --- Tokenize and partition ---

initVocabulary();
const tokens = tokenize(watText as string);

const unknowns = new Array<string>();
for (let i = 0; i < tokens.length; i++) {
  if (!vocab.has(tokens[i])) {
    unknowns.push(tokens[i]);
  }
}

Console.error(
  "Corpus: " + tokens.length.toString() + " tokens, " +
  unknowns.length.toString() + " unknown, " +
  (tokens.length - unknowns.length).toString() + " known\n"
);

// --- Train BPE ---

const merges = trainBpe(unknowns, numMerges);

Console.error("Learned " + merges.length.toString() + " merge rules\n");

// --- Write merges to stdout ---

const output = serializeMerges(merges);
writeOutput(output);

// --- Helpers ---

function writeOutput(text: string): void {
  let start: i32 = 0;
  for (let i: i32 = 0; i < text.length; i++) {
    if (text.charCodeAt(i) == 10) {
      Console.write(text.substring(start, i + 1), false);
      start = i + 1;
    }
  }
  if (start < text.length) {
    Console.write(text.substring(start), false);
  }
}
