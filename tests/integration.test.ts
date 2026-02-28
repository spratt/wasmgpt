import { test, expect } from "assemblyscript-unittest-framework/assembly";
import { initVocabulary, vocab, getNextId } from "../src/vocabulary";
import { tokenize } from "../src/lexer";
import { buildBpeVocab, bpeEncodeToken, getBpeNextId } from "../src/bpe";
import {
  initModel, stateDict, params, vocabSize, gpt,
  N_EMBD, N_LAYER, N_HEAD, BLOCK_SIZE,
} from "../src/model";
import { Tensor, backward, crossEntropy, divScalar, tensorSum, concat } from "../src/tensor";

// ===== Integration: tokenize → encode → forward → loss → backward =====

const WAT_SNIPPET = "(module (func $add (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))";

test("integration: tokenize WAT snippet", () => {
  initVocabulary();
  const tokens = tokenize(WAT_SNIPPET);
  expect(tokens.length > 0).equal(true);
});

test("integration: encode tokens to IDs", () => {
  initVocabulary();
  const tokens = tokenize(WAT_SNIPPET);
  const ids = new Array<i32>();
  for (let i: i32 = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (vocab.has(tok)) {
      ids.push(vocab.get(tok));
    }
  }
  expect(ids.length > 0).equal(true);
  // All IDs should be in valid range
  for (let i: i32 = 0; i < ids.length; i++) {
    expect(ids[i] >= 0).equal(true);
    expect(ids[i] < getNextId()).equal(true);
  }
});

test("integration: forward pass + crossEntropy produces finite loss", () => {
  initVocabulary();
  const vs = getNextId();
  initModel(vs);

  const tokens = tokenize(WAT_SNIPPET);
  const ids = new Array<i32>();
  for (let i: i32 = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (vocab.has(tok)) {
      ids.push(vocab.get(tok));
    }
  }

  // Forward pass over first few tokens
  const seqLen: i32 = ids.length < 8 ? ids.length - 1 : 7;
  const cacheKeys = new Array<Array<Tensor>>(N_LAYER);
  const cacheVals = new Array<Array<Tensor>>(N_LAYER);
  for (let li: i32 = 0; li < N_LAYER; li++) {
    cacheKeys[li] = new Array<Tensor>();
    cacheVals[li] = new Array<Tensor>();
  }

  const losses = new Array<Tensor>(seqLen);
  for (let posId: i32 = 0; posId < seqLen; posId++) {
    const tokenId = ids[posId];
    const targetId = ids[posId + 1];
    const logits = gpt(tokenId, posId, cacheKeys, cacheVals);
    losses[posId] = crossEntropy(logits, targetId);
  }

  const loss = divScalar(tensorSum(concat(losses)), f32(seqLen));

  // Loss should be finite and positive
  expect(isFinite(loss.data[0])).equal(true);
  expect(loss.data[0] > f32(0.0)).equal(true);
});

test("integration: backward produces non-zero gradients", () => {
  initVocabulary();
  const vs = getNextId();
  initModel(vs);

  const tokens = tokenize(WAT_SNIPPET);
  const ids = new Array<i32>();
  for (let i: i32 = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (vocab.has(tok)) {
      ids.push(vocab.get(tok));
    }
  }

  // Short sequence for speed
  const seqLen: i32 = 3;
  const cacheKeys = new Array<Array<Tensor>>(N_LAYER);
  const cacheVals = new Array<Array<Tensor>>(N_LAYER);
  for (let li: i32 = 0; li < N_LAYER; li++) {
    cacheKeys[li] = new Array<Tensor>();
    cacheVals[li] = new Array<Tensor>();
  }

  const losses = new Array<Tensor>(seqLen);
  for (let posId: i32 = 0; posId < seqLen; posId++) {
    const tokenId = ids[posId];
    const targetId = ids[posId + 1];
    const logits = gpt(tokenId, posId, cacheKeys, cacheVals);
    losses[posId] = crossEntropy(logits, targetId);
  }

  const loss = divScalar(tensorSum(concat(losses)), f32(seqLen));
  backward(loss);

  // At least some weight gradients should be non-zero
  let anyNonZero = false;
  for (let i: i32 = 0; i < params.length; i++) {
    const p = params[i];
    for (let j: i32 = 0; j < p.grad.length; j++) {
      if (p.grad[j] != f32(0.0)) {
        anyNonZero = true;
        break;
      }
    }
    if (anyNonZero) break;
  }
  expect(anyNonZero).equal(true);
});

test("integration: graph detached after backward", () => {
  initVocabulary();
  const vs = getNextId();
  initModel(vs);

  const tokens = tokenize(WAT_SNIPPET);
  const ids = new Array<i32>();
  for (let i: i32 = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (vocab.has(tok)) {
      ids.push(vocab.get(tok));
    }
  }

  const cacheKeys = new Array<Array<Tensor>>(N_LAYER);
  const cacheVals = new Array<Array<Tensor>>(N_LAYER);
  for (let li: i32 = 0; li < N_LAYER; li++) {
    cacheKeys[li] = new Array<Tensor>();
    cacheVals[li] = new Array<Tensor>();
  }

  const logits = gpt(ids[0], 0, cacheKeys, cacheVals);
  const loss = crossEntropy(logits, ids[1]);
  backward(loss);

  // After backward, loss should have no children (graph detached)
  expect(loss.children.length).equal(0);
});
