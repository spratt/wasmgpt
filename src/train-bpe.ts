// train-bpe: Train BPE merge rules on WAT corpus files.
// Reads one or more WAT files, tokenizes them, partitions tokens into known
// (Pass 1) and unknown, trains BPE on the unknowns, and writes merge rules
// and vocabulary as S-expressions.
//
// Usage:
//   wasmtime --dir . build/train-bpe.wasm <corpus1.wat> [corpus2.wat] ...

import { CommandLine, Console, FileSystem, Descriptor } from "as-wasi/assembly";
import { readFileText } from "./io";
import { tokenize } from "./lexer";
import { vocab, getNextId, initVocabulary } from "./vocabulary";
import { trainBpe, serializeMerges, buildBpeVocab, serializeVocab, tokenToChars } from "./bpe";
import { parseConfig, configI32 } from "./config";

// --- Parse CLI args ---

const args = CommandLine.all;

if (args.length < 2) {
  Console.error("Usage: train-bpe <corpus1.wat> [corpus2.wat] ...\n");
  abort();
}

// --- Load configuration ---

const configFd = FileSystem.open("config.sexp", "r");
if (configFd === null) {
  Console.error("Error: could not open config.sexp\n");
  abort();
}
const configText = readFileText(configFd as Descriptor);
if (configText === null) {
  Console.error("Error: could not read config.sexp\n");
  abort();
}
const config = parseConfig(configText as string);

// --- Tokenize and partition all corpus files ---

initVocabulary();

let totalTokens: i32 = 0;
const unknowns = new Array<string>();

for (let a: i32 = 1; a < args.length; a++) {
  const corpusPath = args[a];
  const fd = FileSystem.open(corpusPath, "r");
  if (fd === null) {
    Console.error("Error: could not open file: " + corpusPath + "\n");
    abort();
  }
  const watText = readFileText(fd as Descriptor);
  if (watText === null) {
    Console.error("Error: could not read file: " + corpusPath + "\n");
    abort();
  }

  const tokens = tokenize(watText as string);
  totalTokens += tokens.length;

  for (let i: i32 = 0; i < tokens.length; i++) {
    if (!vocab.has(tokens[i])) {
      unknowns.push(tokens[i]);
    }
  }

  Console.error(corpusPath + ": " + tokens.length.toString() + " tokens\n");
}

Console.error(
  "Total: " + totalTokens.toString() + " tokens (" +
  (totalTokens - unknowns.length).toString() + " Pass 1 vocab, " +
  unknowns.length.toString() + " identifiers/literals for BPE)\n"
);

// --- Train BPE ---

const numMerges: i32 = configI32(config, "num-merges", 256);
const merges = trainBpe(unknowns, numMerges);

Console.error("Learned " + merges.length.toString() + " merge rules\n");

// --- Write merges to build/merges.sexp ---

const mergesSexp = serializeMerges(merges);
const mergesFd = FileSystem.open("build/merges.sexp", "w");
if (mergesFd === null) {
  Console.error("Error: could not open build/merges.sexp for writing\n");
  abort();
}
(mergesFd as Descriptor).writeString(mergesSexp);
Console.error("Wrote build/merges.sexp\n");

// --- Build BPE vocab and write vocab.sexp ---

// Collect all unique characters from unknown tokens so every character
// gets a BPE ID, even if it never participated in a merge rule.
const allChars = new Array<string>();
const charSeen = new Map<string, bool>();
for (let i: i32 = 0; i < unknowns.length; i++) {
  const chars = tokenToChars(unknowns[i]);
  for (let j: i32 = 0; j < chars.length; j++) {
    if (!charSeen.has(chars[j])) {
      charSeen.set(chars[j], true);
      allChars.push(chars[j]);
    }
  }
}

buildBpeVocab(merges, getNextId(), allChars);

const vocabSexp = serializeVocab();
const vocabFd = FileSystem.open("build/vocab.sexp", "w");
if (vocabFd === null) {
  Console.error("Error: could not open build/vocab.sexp for writing\n");
  abort();
}
(vocabFd as Descriptor).writeString(vocabSexp);
Console.error("Wrote build/vocab.sexp\n");
