// Pass 2 BPE tokenizer for WasmGPT.
// Learns subword merges for tokens not in the Pass 1 vocabulary.
// Ported from consgpt.lisp's bpe.lisp.

import { nextId } from "./vocabulary";

// Re-export nextId so we can update it from here
export { nextId };

// ===== State =====

export let bpeMerges: Array<Array<string>> = [];
export let bpeVocab: Map<string, i32> = new Map<string, i32>();
export let unkId: i32 = -1;

// Local copy of nextId that we increment during buildBpeVocab.
// We need this because AS doesn't allow re-exporting a mutable binding.
let bpeNextId: i32 = 0;

// ===== Character splitting =====

export function tokenToChars(token: string): Array<string> {
  const result = new Array<string>();
  for (let i = 0; i < token.length; i++) {
    result.push(token.charAt(i));
  }
  return result;
}

// ===== Merge primitives =====

export function mergePair(left: string, right: string, word: Array<string>): Array<string> {
  const result = new Array<string>();
  let i = 0;
  while (i < word.length) {
    if (i + 1 < word.length && word[i] == left && word[i + 1] == right) {
      result.push(left + right);
      i += 2;
    } else {
      result.push(word[i]);
      i++;
    }
  }
  return result;
}

export function countPairs(wordFreqs: Map<string, i32>): Map<string, i32> {
  const pairs = new Map<string, i32>();
  const keys = wordFreqs.keys();
  for (let k = 0; k < keys.length; k++) {
    const wordKey = keys[k];
    const freq = wordFreqs.get(wordKey);
    // Decode word from null-separated key
    const word = wordKey.split("\x01");
    for (let i = 0; i + 1 < word.length; i++) {
      const pairKey = word[i] + "\0" + word[i + 1];
      const existing = pairs.has(pairKey) ? pairs.get(pairKey) : 0;
      pairs.set(pairKey, existing + freq);
    }
  }
  return pairs;
}

// ===== Helpers =====

function encodeWord(word: Array<string>): string {
  let result = "";
  for (let i = 0; i < word.length; i++) {
    if (i > 0) result += "\x01";
    result += word[i];
  }
  return result;
}

// ===== Training =====

export function trainBpe(unknownTokens: Array<string>, numMerges: i32): Array<Array<string>> {
  // Build frequency table: encoded word -> count
  const wordFreqs = new Map<string, i32>();
  for (let i = 0; i < unknownTokens.length; i++) {
    const chars = tokenToChars(unknownTokens[i]);
    if (chars.length == 0) continue;
    const key = encodeWord(chars);
    const existing = wordFreqs.has(key) ? wordFreqs.get(key) : 0;
    wordFreqs.set(key, existing + 1);
  }

  const merges = new Array<Array<string>>();

  for (let step = 0; step < numMerges; step++) {
    const pairs = countPairs(wordFreqs);

    // Find most frequent pair
    let bestPairKey = "";
    let bestCount: i32 = 0;
    const pairKeys = pairs.keys();
    for (let i = 0; i < pairKeys.length; i++) {
      const pk = pairKeys[i];
      const count = pairs.get(pk);
      if (count > bestCount) {
        bestPairKey = pk;
        bestCount = count;
      }
    }

    if (bestCount == 0) break;

    // Decode pair key
    const sepIdx = bestPairKey.indexOf("\0");
    const left = bestPairKey.substring(0, sepIdx);
    const right = bestPairKey.substring(sepIdx + 1);

    merges.push([left, right]);

    // Apply merge to all words
    const newFreqs = new Map<string, i32>();
    const wfKeys = wordFreqs.keys();
    for (let i = 0; i < wfKeys.length; i++) {
      const wk = wfKeys[i];
      const freq = wordFreqs.get(wk);
      const word = wk.split("\x01");
      const merged = mergePair(left, right, word);
      const newKey = encodeWord(merged);
      const existing = newFreqs.has(newKey) ? newFreqs.get(newKey) : 0;
      newFreqs.set(newKey, existing + freq);
    }

    // Replace wordFreqs entries
    const oldKeys = wordFreqs.keys();
    for (let i = 0; i < oldKeys.length; i++) {
      wordFreqs.delete(oldKeys[i]);
    }
    const nfKeys = newFreqs.keys();
    for (let i = 0; i < nfKeys.length; i++) {
      wordFreqs.set(nfKeys[i], newFreqs.get(nfKeys[i]));
    }
  }

  return merges;
}

// ===== Encoding =====

export function applyMerges(chars: Array<string>, merges: Array<Array<string>>): Array<string> {
  let current = chars;
  for (let i = 0; i < merges.length; i++) {
    const merge = merges[i];
    current = mergePair(merge[0], merge[1], current);
  }
  return current;
}

export function bpeEncodeToken(token: string): Array<i32> {
  const chars = tokenToChars(token);
  if (chars.length == 0) return new Array<i32>();
  const subwords = applyMerges(chars, bpeMerges);
  const ids = new Array<i32>();
  for (let i = 0; i < subwords.length; i++) {
    const sw = subwords[i];
    if (bpeVocab.has(sw)) {
      ids.push(bpeVocab.get(sw));
    } else {
      ids.push(unkId);
    }
  }
  return ids;
}

// ===== Vocabulary building =====

function stringLessThan(a: string, b: string): i32 {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

export function buildBpeVocab(merges: Array<Array<string>>, startId: i32): void {
  bpeMerges = merges;
  bpeVocab = new Map<string, i32>();
  bpeNextId = startId;

  // Collect all unique characters from merge rules
  const charSet = new Map<string, bool>();
  for (let i = 0; i < merges.length; i++) {
    const merge = merges[i];
    const leftChars = tokenToChars(merge[0]);
    for (let j = 0; j < leftChars.length; j++) {
      charSet.set(leftChars[j], true);
    }
    const rightChars = tokenToChars(merge[1]);
    for (let j = 0; j < rightChars.length; j++) {
      charSet.set(rightChars[j], true);
    }
  }

  // Register single characters first, sorted
  const sortedChars = charSet.keys();
  sortedChars.sort(stringLessThan);
  for (let i = 0; i < sortedChars.length; i++) {
    const ch = sortedChars[i];
    if (!bpeVocab.has(ch)) {
      bpeVocab.set(ch, bpeNextId);
      bpeNextId++;
    }
  }

  // Register merged symbols in merge order
  for (let i = 0; i < merges.length; i++) {
    const merged = merges[i][0] + merges[i][1];
    if (!bpeVocab.has(merged)) {
      bpeVocab.set(merged, bpeNextId);
      bpeNextId++;
    }
  }

  // UNK fallback
  unkId = bpeNextId;
  bpeVocab.set("<UNK>", unkId);
  bpeNextId++;
}

export function getBpeNextId(): i32 {
  return bpeNextId;
}
