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

// --- S-expression serialization format ---
// These tests verify that the lexer can parse the S-expression format
// we plan to use for BPE merges and vocabulary persistence.

// Merges format: (merges ("a" "b") ("st" "ar") ...)

test("sexp: simple merge pair", () => {
  const result = tokenize('("a" "b")');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"a"');
  expect(result[2]).equal('"b"');
  expect(result[3]).equal(")");
});

test("sexp: merge list with multiple pairs", () => {
  const result = tokenize('(merges ("s" "t") ("st" "ar"))');
  expect(result.length).equal(11);
  expect(result[0]).equal("(");
  expect(result[1]).equal("merges");
  expect(result[2]).equal("(");
  expect(result[3]).equal('"s"');
  expect(result[4]).equal('"t"');
  expect(result[5]).equal(")");
  expect(result[6]).equal("(");
  expect(result[7]).equal('"st"');
  expect(result[8]).equal('"ar"');
  expect(result[9]).equal(")");
  expect(result[10]).equal(")");
});

// Vocab format: (vocab ("(" 0) ("module" 2) ...)

test("sexp: vocab entry with integer ID", () => {
  const result = tokenize('("module" 2)');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"module"');
  expect(result[2]).equal("2");
  expect(result[3]).equal(")");
});

test("sexp: vocab list", () => {
  const result = tokenize('(vocab ("(" 0) (")" 1) ("module" 2))');
  expect(result.length).equal(15);
  expect(result[0]).equal("(");
  expect(result[1]).equal("vocab");
  expect(result[2]).equal("(");
  expect(result[3]).equal('"("');
  expect(result[4]).equal("0");
  expect(result[5]).equal(")");
  expect(result[6]).equal("(");
  expect(result[7]).equal('")"');
  expect(result[8]).equal("1");
  expect(result[9]).equal(")");
  expect(result[10]).equal("(");
  expect(result[11]).equal('"module"');
  expect(result[12]).equal("2");
  expect(result[13]).equal(")");
  expect(result[14]).equal(")");
});

// Edge cases: tokens that caused the TSV bug

test("sexp: string containing backslash-zero", () => {
  const result = tokenize('("\\00" "x")');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"\\00"');
  expect(result[2]).equal('"x"');
  expect(result[3]).equal(")");
});

test("sexp: string containing backslash-n", () => {
  const result = tokenize('("\\n" "y")');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"\\n"');
  expect(result[2]).equal('"y"');
  expect(result[3]).equal(")");
});

test("sexp: string containing backslash-t", () => {
  const result = tokenize('("\\t" "z")');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"\\t"');
  expect(result[2]).equal('"z"');
  expect(result[3]).equal(")");
});

test("sexp: string containing escaped quote", () => {
  const result = tokenize('("\\"" 99)');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"\\""');
  expect(result[2]).equal("99");
  expect(result[3]).equal(")");
});

test("sexp: string containing backslash-backslash", () => {
  const result = tokenize('("\\\\" 50)');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"\\\\"');
  expect(result[2]).equal("50");
  expect(result[3]).equal(")");
});

test("sexp: multiple null-escape merge pairs", () => {
  // A bare backslash must be escaped as \\\\ in the S-expression
  const result = tokenize('(merges ("\\\\" "0") ("\\0" "0") ("\\00" "\\00"))');
  expect(result.length).equal(15);
  expect(result[0]).equal("(");
  expect(result[1]).equal("merges");
  // ("\\\\" "0") — backslash token
  expect(result[2]).equal("(");
  expect(result[3]).equal('"\\\\"');
  expect(result[4]).equal('"0"');
  expect(result[5]).equal(")");
  // ("\\0" "0")
  expect(result[6]).equal("(");
  expect(result[7]).equal('"\\0"');
  expect(result[8]).equal('"0"');
  expect(result[9]).equal(")");
  // ("\\00" "\\00")
  expect(result[10]).equal("(");
  expect(result[11]).equal('"\\00"');
  expect(result[12]).equal('"\\00"');
  expect(result[13]).equal(")");
});

test("sexp: token containing parens as string", () => {
  const result = tokenize('("(" 0)');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"("');
  expect(result[2]).equal("0");
  expect(result[3]).equal(")");
});

test("sexp: token containing semicolons as string", () => {
  const result = tokenize('(";;" 42)');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('";;"');
  expect(result[2]).equal("42");
  expect(result[3]).equal(")");
});

test("sexp: multiline format", () => {
  const result = tokenize('(merges\n  ("a" "b")\n  ("c" "d")\n)');
  expect(result.length).equal(11);
  expect(result[0]).equal("(");
  expect(result[1]).equal("merges");
  expect(result[2]).equal("(");
  expect(result[3]).equal('"a"');
  expect(result[4]).equal('"b"');
  expect(result[5]).equal(")");
  expect(result[6]).equal("(");
  expect(result[7]).equal('"c"');
  expect(result[8]).equal('"d"');
  expect(result[9]).equal(")");
  expect(result[10]).equal(")");
});

test("sexp: real BPE token $~lib/memory/__stack_pointer", () => {
  const result = tokenize('("$~lib/memory/__stack_pointer" 100)');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"$~lib/memory/__stack_pointer"');
  expect(result[2]).equal("100");
  expect(result[3]).equal(")");
});

test("sexp: empty string token", () => {
  const result = tokenize('("" 0)');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('""');
  expect(result[2]).equal("0");
  expect(result[3]).equal(")");
});

test("sexp: string with spaces", () => {
  const result = tokenize('("hello world" 5)');
  expect(result.length).equal(4);
  expect(result[0]).equal("(");
  expect(result[1]).equal('"hello world"');
  expect(result[2]).equal("5");
  expect(result[3]).equal(")");
});
