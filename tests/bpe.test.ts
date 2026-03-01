import { test, expect } from "assemblyscript-unittest-framework/assembly";
import {
  tokenToChars, mergePair, countPairs,
  trainBpe, applyMerges, bpeEncodeToken,
  buildBpeVocab, getBpeNextId,
  bpeVocab, unkId,
  serializeMerges, parseMerges, parseVocab,
  escapeForSexp, unescapeFromSexp, stripQuotes,
  serializeVocab,
} from "../src/bpe";
import { initVocabulary, getNextId } from "../src/vocabulary";

// ===== tokenToChars =====

test("tokenToChars normal", () => {
  const result = tokenToChars("HELLO");
  expect(result.length).equal(5);
  expect(result[0]).equal("H");
  expect(result[1]).equal("E");
  expect(result[2]).equal("L");
  expect(result[3]).equal("L");
  expect(result[4]).equal("O");
});

test("tokenToChars single char", () => {
  const result = tokenToChars("X");
  expect(result.length).equal(1);
  expect(result[0]).equal("X");
});

test("tokenToChars empty", () => {
  const result = tokenToChars("");
  expect(result.length).equal(0);
});

// ===== mergePair =====

test("mergePair basic", () => {
  const result = mergePair("A", "B", ["A", "B", "C"]);
  expect(result.length).equal(2);
  expect(result[0]).equal("AB");
  expect(result[1]).equal("C");
});

test("mergePair no match", () => {
  const result = mergePair("X", "Y", ["A", "C"]);
  expect(result.length).equal(2);
  expect(result[0]).equal("A");
  expect(result[1]).equal("C");
});

test("mergePair multiple occurrences", () => {
  const result = mergePair("A", "B", ["A", "B", "C", "A", "B"]);
  expect(result.length).equal(3);
  expect(result[0]).equal("AB");
  expect(result[1]).equal("C");
  expect(result[2]).equal("AB");
});

test("mergePair single element", () => {
  const result = mergePair("A", "B", ["A"]);
  expect(result.length).equal(1);
  expect(result[0]).equal("A");
});

// ===== countPairs =====

test("countPairs weighted frequency", () => {
  const wordFreqs = new Map<string, i32>();
  // "A","B","C" with freq 3 and "A","B" with freq 2
  wordFreqs.set("A\x01B\x01C", 3);
  wordFreqs.set("A\x01B", 2);
  const pairs = countPairs(wordFreqs);
  // AB appears in both: 3 + 2 = 5
  const abKey = "A\0B";
  expect(pairs.has(abKey)).equal(true);
  expect(pairs.get(abKey)).equal(5);
  // BC appears only in first: 3
  const bcKey = "B\0C";
  expect(pairs.has(bcKey)).equal(true);
  expect(pairs.get(bcKey)).equal(3);
});

// ===== applyMerges =====

test("applyMerges single merge", () => {
  const merges: Array<Array<string>> = [["A", "B"]];
  const result = applyMerges(["A", "B", "C"], merges);
  expect(result.length).equal(2);
  expect(result[0]).equal("AB");
  expect(result[1]).equal("C");
});

test("applyMerges chained", () => {
  const merges: Array<Array<string>> = [["A", "B"], ["AB", "C"]];
  const result = applyMerges(["A", "B", "C"], merges);
  expect(result.length).equal(1);
  expect(result[0]).equal("ABC");
});

test("applyMerges no applicable merges", () => {
  const merges: Array<Array<string>> = [["A", "B"]];
  const result = applyMerges(["X", "Y"], merges);
  expect(result.length).equal(2);
  expect(result[0]).equal("X");
  expect(result[1]).equal("Y");
});

// ===== trainBpe =====

test("trainBpe empty input", () => {
  const merges = trainBpe([], 10);
  expect(merges.length).equal(0);
});

test("trainBpe single-char tokens", () => {
  const merges = trainBpe(["A", "B", "C"], 10);
  expect(merges.length).equal(0);
});

test("trainBpe learns most frequent pair first", () => {
  const tokens: Array<string> = ["AB", "AB", "AB", "CD", "CD"];
  const merges = trainBpe(tokens, 1);
  expect(merges.length).equal(1);
  expect(merges[0][0]).equal("A");
  expect(merges[0][1]).equal("B");
});

test("trainBpe returns list of pairs", () => {
  const tokens: Array<string> = ["DEFUN", "DEFUN", "DEFUN", "DEFVAR", "DEFVAR"];
  const merges = trainBpe(tokens, 10);
  expect(merges.length > 0).equal(true);
  for (let i = 0; i < merges.length; i++) {
    expect(merges[i].length).equal(2);
  }
});

// ===== buildBpeVocab =====

test("buildBpeVocab registers chars", () => {
  const merges: Array<Array<string>> = [["A", "B"], ["AB", "C"]];
  buildBpeVocab(merges, 100);
  expect(bpeVocab.has("A")).equal(true);
  expect(bpeVocab.has("B")).equal(true);
  expect(bpeVocab.has("C")).equal(true);
});

test("buildBpeVocab registers merged symbols", () => {
  const merges: Array<Array<string>> = [["A", "B"], ["AB", "C"]];
  buildBpeVocab(merges, 100);
  expect(bpeVocab.has("AB")).equal(true);
  expect(bpeVocab.has("ABC")).equal(true);
});

test("buildBpeVocab sets unkId", () => {
  const merges: Array<Array<string>> = [["A", "B"], ["AB", "C"]];
  buildBpeVocab(merges, 100);
  expect(unkId >= 0).equal(true);
  expect(bpeVocab.has("<UNK>")).equal(true);
  expect(bpeVocab.get("<UNK>")).equal(unkId);
});

test("buildBpeVocab IDs start after vocabulary nextId", () => {
  initVocabulary();
  const startId = getNextId();
  const merges: Array<Array<string>> = [["A", "B"]];
  buildBpeVocab(merges, startId);
  // A and B are chars, AB is merged, <UNK> is last
  expect(bpeVocab.get("A")).equal(startId);
  expect(bpeVocab.get("B")).equal(startId + 1);
  expect(bpeVocab.get("AB")).equal(startId + 2);
  expect(bpeVocab.get("<UNK>")).equal(startId + 3);
});

test("buildBpeVocab chars are sorted", () => {
  const merges: Array<Array<string>> = [["C", "A"], ["B", "D"]];
  buildBpeVocab(merges, 0);
  // Sorted chars: A, B, C, D → IDs 0, 1, 2, 3
  expect(bpeVocab.get("A")).equal(0);
  expect(bpeVocab.get("B")).equal(1);
  expect(bpeVocab.get("C")).equal(2);
  expect(bpeVocab.get("D")).equal(3);
});

test("buildBpeVocab UNK is last", () => {
  const merges: Array<Array<string>> = [["A", "B"]];
  buildBpeVocab(merges, 0);
  const totalIds = getBpeNextId();
  expect(unkId).equal(totalIds - 1);
});

// ===== bpeEncodeToken =====

test("bpeEncodeToken mergeable token", () => {
  const merges: Array<Array<string>> = [["A", "B"], ["AB", "C"]];
  buildBpeVocab(merges, 0);
  const ids = bpeEncodeToken("ABC");
  expect(ids.length).equal(1);
  expect(ids[0]).equal(bpeVocab.get("ABC"));
});

test("bpeEncodeToken single char", () => {
  const merges: Array<Array<string>> = [["A", "B"], ["AB", "C"]];
  buildBpeVocab(merges, 0);
  const ids = bpeEncodeToken("A");
  expect(ids.length).equal(1);
  expect(ids[0]).equal(bpeVocab.get("A"));
});

test("bpeEncodeToken unknown char without corpusChars", () => {
  const merges: Array<Array<string>> = [["A", "B"], ["AB", "C"]];
  buildBpeVocab(merges, 0);
  const ids = bpeEncodeToken("Z");
  expect(ids.length).equal(1);
  expect(ids[0]).equal(unkId);
});

test("bpeEncodeToken known char via corpusChars", () => {
  const merges: Array<Array<string>> = [["A", "B"], ["AB", "C"]];
  buildBpeVocab(merges, 0, ["Z"]);
  const ids = bpeEncodeToken("Z");
  expect(ids.length).equal(1);
  expect(ids[0] != unkId).equal(true);
  expect(bpeVocab.has("Z")).equal(true);
});

test("bpeEncodeToken empty token", () => {
  const merges: Array<Array<string>> = [["A", "B"]];
  buildBpeVocab(merges, 0);
  const ids = bpeEncodeToken("");
  expect(ids.length).equal(0);
});

// ===== escapeForSexp / unescapeFromSexp / stripQuotes =====

test("escapeForSexp escapes backslash", () => {
  expect(escapeForSexp("a\\b")).equal("a\\\\b");
});

test("escapeForSexp escapes quote", () => {
  expect(escapeForSexp('say "hi"')).equal('say \\"hi\\"');
});

test("escapeForSexp no-op for plain string", () => {
  expect(escapeForSexp("hello")).equal("hello");
});

test("unescapeFromSexp reverses backslash", () => {
  expect(unescapeFromSexp("a\\\\b")).equal("a\\b");
});

test("unescapeFromSexp reverses quote", () => {
  expect(unescapeFromSexp('say \\"hi\\"')).equal('say "hi"');
});

test("stripQuotes removes surrounding quotes", () => {
  expect(stripQuotes('"hello"')).equal("hello");
});

test("stripQuotes returns unquoted string unchanged", () => {
  expect(stripQuotes("hello")).equal("hello");
});

// ===== serializeMerges / parseMerges =====

test("serializeMerges produces S-expression", () => {
  const merges: Array<Array<string>> = [["A", "B"], ["AB", "C"]];
  const result = serializeMerges(merges);
  expect(result).equal('(merges\n  ("A" "B")\n  ("AB" "C"))\n');
});

test("serializeMerges empty merges", () => {
  const merges = new Array<Array<string>>();
  const result = serializeMerges(merges);
  expect(result).equal("(merges)\n");
});

test("serializeMerges escapes backslash tokens", () => {
  const merges: Array<Array<string>> = [["\\", "0"]];
  const result = serializeMerges(merges);
  expect(result).equal('(merges\n  ("\\\\" "0"))\n');
});

test("parseMerges round-trip", () => {
  const original: Array<Array<string>> = [["A", "B"], ["AB", "C"], ["$", "a"]];
  const serialized = serializeMerges(original);
  const parsed = parseMerges(serialized);
  expect(parsed.length).equal(3);
  expect(parsed[0][0]).equal("A");
  expect(parsed[0][1]).equal("B");
  expect(parsed[1][0]).equal("AB");
  expect(parsed[1][1]).equal("C");
  expect(parsed[2][0]).equal("$");
  expect(parsed[2][1]).equal("a");
});

test("parseMerges round-trip with backslash and quote tokens", () => {
  const original: Array<Array<string>> = [["\\", "0"], ["\\0", "0"], ["\"", "x"]];
  const serialized = serializeMerges(original);
  const parsed = parseMerges(serialized);
  expect(parsed.length).equal(3);
  expect(parsed[0][0]).equal("\\");
  expect(parsed[0][1]).equal("0");
  expect(parsed[1][0]).equal("\\0");
  expect(parsed[1][1]).equal("0");
  expect(parsed[2][0]).equal("\"");
  expect(parsed[2][1]).equal("x");
});

test("parseMerges with \\00 tokens like real merges file", () => {
  const original: Array<Array<string>> = [
    ["\\", "0"],
    ["\\0", "0"],
    ["\\00", "\\00"],
    ["\\00\\00", "\\00\\00"],
    ["l", "o"],
    ["lo", "cal"],
  ];
  const serialized = serializeMerges(original);
  const parsed = parseMerges(serialized);
  expect(parsed.length).equal(6);
  expect(parsed[0][0]).equal("\\");
  expect(parsed[0][1]).equal("0");
  expect(parsed[1][0]).equal("\\0");
  expect(parsed[1][1]).equal("0");
  expect(parsed[2][0]).equal("\\00");
  expect(parsed[2][1]).equal("\\00");
  expect(parsed[3][0]).equal("\\00\\00");
  expect(parsed[3][1]).equal("\\00\\00");
  expect(parsed[4][0]).equal("l");
  expect(parsed[4][1]).equal("o");
  expect(parsed[5][0]).equal("lo");
  expect(parsed[5][1]).equal("cal");
});

test("parseMerges with many entries", () => {
  const original = new Array<Array<string>>();
  for (let i = 0; i < 100; i++) {
    original.push(["a" + i.toString(), "b" + i.toString()]);
  }
  const serialized = serializeMerges(original);
  const parsed = parseMerges(serialized);
  expect(parsed.length).equal(100);
  expect(parsed[0][0]).equal("a0");
  expect(parsed[99][0]).equal("a99");
});

test("parseMerges round-trip reproduces actual first merges", () => {
  // Reproduce the exact first entries from the real merges.sexp
  const original: Array<Array<string>> = [
    ["\\", "0"],
    ["\\0", "0"],
    ["\\00", "\\00"],
    ["\\00\\00", "\\00\\00"],
    ["l", "o"],
    ["lo", "cal"],
    ["i", "3"],
    ["i3", "2"],
    [",", "\\00"],
    ["$", "~"],
  ];
  const serialized = serializeMerges(original);
  const parsed = parseMerges(serialized);
  expect(parsed.length).equal(10);
  for (let i = 0; i < 10; i++) {
    expect(parsed[i][0]).equal(original[i][0]);
    expect(parsed[i][1]).equal(original[i][1]);
  }
});

// ===== parseVocab =====

test("parseVocab parses S-expression vocab", () => {
  const text = '(vocab\n  ("(" 0)\n  (")" 1)\n  ("module" 2))\n';
  const result = parseVocab(text);
  expect(result.size).equal(3);
  expect(result.get("(")).equal(0);
  expect(result.get(")")).equal(1);
  expect(result.get("module")).equal(2);
});

test("parseVocab handles escaped tokens", () => {
  const text = '(vocab\n  ("\\\\" 10)\n  ("\\"" 11))\n';
  const result = parseVocab(text);
  expect(result.size).equal(2);
  expect(result.get("\\")).equal(10);
  expect(result.get("\"")).equal(11);
});
