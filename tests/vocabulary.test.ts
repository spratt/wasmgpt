import { test, expect } from "assemblyscript-unittest-framework/assembly";
import { vocab, nextId, initVocabulary } from "../src/vocabulary";

// --- Initialization ---

test("initVocabulary populates vocab", () => {
  initVocabulary();
  expect(vocab.size > 0).equal(true);
});

test("nextId equals vocab size", () => {
  initVocabulary();
  expect(nextId).equal(vocab.size);
});

// --- Syntax tokens ---

test("open paren has ID 0", () => {
  initVocabulary();
  expect(vocab.has("(")).equal(true);
  expect(vocab.get("(")).equal(0);
});

test("close paren has ID 1", () => {
  initVocabulary();
  expect(vocab.has(")")).equal(true);
  expect(vocab.get(")")).equal(1);
});

// --- Structural keywords ---

test("module is present", () => {
  initVocabulary();
  expect(vocab.has("module")).equal(true);
});

test("func is present", () => {
  initVocabulary();
  expect(vocab.has("func")).equal(true);
});

test("param is present", () => {
  initVocabulary();
  expect(vocab.has("param")).equal(true);
});

test("result is present", () => {
  initVocabulary();
  expect(vocab.has("result")).equal(true);
});

test("mut is present", () => {
  initVocabulary();
  expect(vocab.has("mut")).equal(true);
});

test("import is present", () => {
  initVocabulary();
  expect(vocab.has("import")).equal(true);
});

test("export is present", () => {
  initVocabulary();
  expect(vocab.has("export")).equal(true);
});

// --- Type names ---

test("i32 type is present", () => {
  initVocabulary();
  expect(vocab.has("i32")).equal(true);
});

test("i64 type is present", () => {
  initVocabulary();
  expect(vocab.has("i64")).equal(true);
});

test("f32 type is present", () => {
  initVocabulary();
  expect(vocab.has("f32")).equal(true);
});

test("f64 type is present", () => {
  initVocabulary();
  expect(vocab.has("f64")).equal(true);
});

test("v128 type is present", () => {
  initVocabulary();
  expect(vocab.has("v128")).equal(true);
});

test("funcref type is present", () => {
  initVocabulary();
  expect(vocab.has("funcref")).equal(true);
});

test("externref type is present", () => {
  initVocabulary();
  expect(vocab.has("externref")).equal(true);
});

// --- Known instruction mnemonics ---

test("i32.add is present", () => {
  initVocabulary();
  expect(vocab.has("i32.add")).equal(true);
});

test("local.get is present", () => {
  initVocabulary();
  expect(vocab.has("local.get")).equal(true);
});

test("call is present", () => {
  initVocabulary();
  expect(vocab.has("call")).equal(true);
});

test("nop is present", () => {
  initVocabulary();
  expect(vocab.has("nop")).equal(true);
});

test("f64.mul is present", () => {
  initVocabulary();
  expect(vocab.has("f64.mul")).equal(true);
});

test("v128.load is present", () => {
  initVocabulary();
  expect(vocab.has("v128.load")).equal(true);
});

test("i32.atomic.rmw.add is present", () => {
  initVocabulary();
  expect(vocab.has("i32.atomic.rmw.add")).equal(true);
});

test("i8x16.shuffle is present", () => {
  initVocabulary();
  expect(vocab.has("i8x16.shuffle")).equal(true);
});

// --- No duplicate IDs ---

test("no duplicate IDs", () => {
  initVocabulary();
  const ids = new Set<i32>();
  const keys = vocab.keys();
  for (let i = 0; i < keys.length; i++) {
    const id = vocab.get(keys[i]);
    ids.add(id);
  }
  expect(ids.size).equal(vocab.size);
});

// --- IDs are sequential from 0 ---

test("IDs range from 0 to nextId-1", () => {
  initVocabulary();
  const keys = vocab.keys();
  let minId: i32 = i32.MAX_VALUE;
  let maxId: i32 = i32.MIN_VALUE;
  for (let i = 0; i < keys.length; i++) {
    const id = vocab.get(keys[i]);
    if (id < minId) minId = id;
    if (id > maxId) maxId = id;
  }
  expect(minId).equal(0);
  expect(maxId).equal(nextId - 1);
});

// --- Idempotency ---

test("calling initVocabulary twice does not change IDs", () => {
  initVocabulary();
  const firstSize = vocab.size;
  const firstNextId = nextId;
  const addId = vocab.get("i32.add");

  initVocabulary();
  expect(vocab.size).equal(firstSize);
  expect(nextId).equal(firstNextId);
  expect(vocab.get("i32.add")).equal(addId);
});

// --- Structural keywords have lower IDs than instructions ---

test("structural keywords have IDs before instructions", () => {
  initVocabulary();
  const moduleId = vocab.get("module");
  const addId = vocab.get("i32.add");
  expect(moduleId < addId).equal(true);
});

// --- Overlapping keywords are handled ---

test("block is registered (both keyword and instruction)", () => {
  initVocabulary();
  expect(vocab.has("block")).equal(true);
});

test("select is registered (appears twice in opcode.def)", () => {
  initVocabulary();
  expect(vocab.has("select")).equal(true);
});
