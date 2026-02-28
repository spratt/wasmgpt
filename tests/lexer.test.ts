import { test, expect } from "assemblyscript-unittest-framework/assembly";
import { tokenize } from "../src/lexer";

// --- Empty / whitespace ---

test("empty input", () => {
  const result = tokenize("");
  expect(result.length).equal(0);
});

test("whitespace only", () => {
  const result = tokenize("   \n\t  ");
  expect(result.length).equal(0);
});

// --- Parentheses ---

test("open paren", () => {
  const result = tokenize("(");
  expect(result.length).equal(1);
  expect(result[0]).equal("(");
});

test("close paren", () => {
  const result = tokenize(")");
  expect(result.length).equal(1);
  expect(result[0]).equal(")");
});

test("nested parens", () => {
  const result = tokenize("(())");
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal("(");
  expect(result[2]).equal(")");
  expect(result[3]).equal(")");
});

// --- Atoms: instruction mnemonics ---

test("simple mnemonic", () => {
  const result = tokenize("nop");
  expect(result.length).equal(1);
  expect(result[0]).equal("nop");
});

test("dotted mnemonic", () => {
  const result = tokenize("i32.add");
  expect(result.length).equal(1);
  expect(result[0]).equal("i32.add");
});

test("mnemonic in parens", () => {
  const result = tokenize("(i32.const 42)");
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal("i32.const");
  expect(result[2]).equal("42");
  expect(result[3]).equal(")");
});

// --- Atoms: identifiers ---

test("dollar identifier", () => {
  const result = tokenize("$func_name");
  expect(result.length).equal(1);
  expect(result[0]).equal("$func_name");
});

test("identifier with path chars", () => {
  const result = tokenize("$~lib/rt/itcms/state");
  expect(result.length).equal(1);
  expect(result[0]).equal("$~lib/rt/itcms/state");
});

// --- Atoms: numeric literals ---

test("integer literal", () => {
  const result = tokenize("42");
  expect(result.length).equal(1);
  expect(result[0]).equal("42");
});

test("hex literal", () => {
  const result = tokenize("0xFF");
  expect(result.length).equal(1);
  expect(result[0]).equal("0xFF");
});

test("negative integer", () => {
  const result = tokenize("-1");
  expect(result.length).equal(1);
  expect(result[0]).equal("-1");
});

test("float literal", () => {
  const result = tokenize("0.5");
  expect(result.length).equal(1);
  expect(result[0]).equal("0.5");
});

// --- Atoms: keywords and types ---

test("keywords are case-sensitive", () => {
  const result = tokenize("module func param result");
  expect(result.length).equal(4);
  expect(result[0]).equal("module");
  expect(result[1]).equal("func");
  expect(result[2]).equal("param");
  expect(result[3]).equal("result");
});

test("type names", () => {
  const result = tokenize("i32 i64 f32 f64");
  expect(result.length).equal(4);
  expect(result[0]).equal("i32");
  expect(result[1]).equal("i64");
  expect(result[2]).equal("f32");
  expect(result[3]).equal("f64");
});

// --- String literals ---

test("simple string", () => {
  const result = tokenize('"hello"');
  expect(result.length).equal(1);
  expect(result[0]).equal('"hello"');
});

test("empty string", () => {
  const result = tokenize('""');
  expect(result.length).equal(1);
  expect(result[0]).equal('""');
});

test("string with escape sequences", () => {
  const result = tokenize('"\\01\\02\\03"');
  expect(result.length).equal(1);
  expect(result[0]).equal('"\\01\\02\\03"');
});

test("string with escaped quote", () => {
  const result = tokenize('"say \\"hi\\""');
  expect(result.length).equal(1);
  expect(result[0]).equal('"say \\"hi\\""');
});

// --- Line comments ---

test("line comment is stripped", () => {
  const result = tokenize(";; this is a comment\nnop");
  expect(result.length).equal(1);
  expect(result[0]).equal("nop");
});

test("line comment after token", () => {
  const result = tokenize("nop ;; comment");
  expect(result.length).equal(1);
  expect(result[0]).equal("nop");
});

test("comment-only input", () => {
  const result = tokenize(";; just a comment");
  expect(result.length).equal(0);
});

// --- Block comments ---

test("block comment is stripped", () => {
  const result = tokenize("(; block comment ;) nop");
  expect(result.length).equal(1);
  expect(result[0]).equal("nop");
});

test("nested block comments", () => {
  const result = tokenize("(; outer (; inner ;) still outer ;) nop");
  expect(result.length).equal(1);
  expect(result[0]).equal("nop");
});

test("block comment between tokens", () => {
  const result = tokenize("( (; comment ;) module)");
  expect(result.length).equal(3);
  expect(result[0]).equal("(");
  expect(result[1]).equal("module");
  expect(result[2]).equal(")");
});

// --- Integration: real WAT snippets ---

test("simple module", () => {
  const wat = '(module\n  (func (export "fac") (param i32) (result i32)\n    (i32.const 1)))';
  const result = tokenize(wat);
  expect(result[0]).equal("(");
  expect(result[1]).equal("module");
  expect(result[2]).equal("(");
  expect(result[3]).equal("func");
  expect(result[4]).equal("(");
  expect(result[5]).equal("export");
  expect(result[6]).equal('"fac"');
  expect(result[7]).equal(")");
  expect(result[8]).equal("(");
  expect(result[9]).equal("param");
  expect(result[10]).equal("i32");
  expect(result[11]).equal(")");
});

test("function with comments", () => {
  const wat = ";; A function\n(func $rot13 (param $c i32) (result i32)\n  ;; body\n  (local.get $c))";
  const result = tokenize(wat);
  expect(result[0]).equal("(");
  expect(result[1]).equal("func");
  expect(result[2]).equal("$rot13");
  expect(result[3]).equal("(");
  expect(result[4]).equal("param");
  expect(result[5]).equal("$c");
  expect(result[6]).equal("i32");
  expect(result[7]).equal(")");
  expect(result[8]).equal("(");
  expect(result[9]).equal("result");
  expect(result[10]).equal("i32");
  expect(result[11]).equal(")");
  expect(result[12]).equal("(");
  expect(result[13]).equal("local.get");
  expect(result[14]).equal("$c");
  expect(result[15]).equal(")");
  expect(result[16]).equal(")");
});

test("global with mut", () => {
  const result = tokenize("(global $x (mut i32) (i32.const 0))");
  expect(result[0]).equal("(");
  expect(result[1]).equal("global");
  expect(result[2]).equal("$x");
  expect(result[3]).equal("(");
  expect(result[4]).equal("mut");
  expect(result[5]).equal("i32");
  expect(result[6]).equal(")");
  expect(result[7]).equal("(");
  expect(result[8]).equal("i32.const");
  expect(result[9]).equal("0");
  expect(result[10]).equal(")");
  expect(result[11]).equal(")");
});

test("data segment with hex and string", () => {
  const result = tokenize('(data (i32.const 0x1000) "\\01\\02")');
  expect(result[0]).equal("(");
  expect(result[1]).equal("data");
  expect(result[2]).equal("(");
  expect(result[3]).equal("i32.const");
  expect(result[4]).equal("0x1000");
  expect(result[5]).equal(")");
  expect(result[6]).equal('"\\01\\02"');
  expect(result[7]).equal(")");
});

// --- Edge cases ---

test("atom directly before comment no space", () => {
  const result = tokenize("nop;;comment");
  expect(result.length).equal(1);
  expect(result[0]).equal("nop");
});

test("multiple spaces between tokens", () => {
  const result = tokenize("i32.add   i32.sub");
  expect(result.length).equal(2);
  expect(result[0]).equal("i32.add");
  expect(result[1]).equal("i32.sub");
});
