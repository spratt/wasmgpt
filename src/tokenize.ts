// wasmgpt: A GPT trained on WebAssembly
// Entry point that exercises the tokenizer pipeline.

import { tokenize } from "./lexer";
import { vocab, getNextId, initVocabulary } from "./vocabulary";
import { buildBpeVocab, bpeEncodeToken, parseMerges, getBpeNextId } from "./bpe";

initVocabulary();

export function getVocabSize(): i32 {
  return getNextId();
}

export function tokenizeWat(text: string): Array<string> {
  return tokenize(text);
}

export function lookupToken(token: string): i32 {
  if (vocab.has(token)) {
    return vocab.get(token);
  }
  return -1;
}

export function loadMerges(mergesTsv: string): void {
  const merges = parseMerges(mergesTsv);
  buildBpeVocab(merges, getNextId());
}

export function encodeToken(token: string): Array<i32> {
  return bpeEncodeToken(token);
}

export function getTotalVocabSize(): i32 {
  return getBpeNextId();
}
