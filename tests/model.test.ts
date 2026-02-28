import { test, expect } from "assemblyscript-unittest-framework/assembly";
import {
  initModel, stateDict, params, numParams, vocabSize, gpt,
  N_EMBD, N_LAYER, N_HEAD, HEAD_DIM, BLOCK_SIZE,
  lcgState, randomGaussian, weightedChoice, detachKvCache,
} from "../src/model";
import { Tensor } from "../src/tensor";

// ===== Hyperparameters =====

test("hyperparameters are correct", () => {
  expect(N_EMBD).equal(64);
  expect(N_LAYER).equal(2);
  expect(N_HEAD).equal(4);
  expect(HEAD_DIM).equal(16);
  expect(BLOCK_SIZE).equal(256);
  expect(N_EMBD).equal(N_HEAD * HEAD_DIM);
});

// ===== PRNG =====

test("randomGaussian produces finite values", () => {
  for (let i: i32 = 0; i < 100; i++) {
    const g = randomGaussian();
    expect(isFinite(g)).equal(true);
  }
});

// ===== Model initialization =====

test("initModel creates 14 weight tensors", () => {
  initModel(100);
  expect(stateDict.size).equal(14);
  expect(params.length).equal(14);
  expect(vocabSize).equal(100);
});

test("initModel creates correct tensor shapes", () => {
  initModel(100);
  const wte = stateDict.get("wte");
  expect(wte.shape[0]).equal(100);
  expect(wte.shape[1]).equal(N_EMBD);

  const wpe = stateDict.get("wpe");
  expect(wpe.shape[0]).equal(BLOCK_SIZE);
  expect(wpe.shape[1]).equal(N_EMBD);

  const wq = stateDict.get("layer0.attn_wq");
  expect(wq.shape[0]).equal(N_EMBD);
  expect(wq.shape[1]).equal(N_EMBD);

  const fc1 = stateDict.get("layer0.mlp_fc1");
  expect(fc1.shape[0]).equal(4 * N_EMBD);
  expect(fc1.shape[1]).equal(N_EMBD);

  const fc2 = stateDict.get("layer0.mlp_fc2");
  expect(fc2.shape[0]).equal(N_EMBD);
  expect(fc2.shape[1]).equal(4 * N_EMBD);
});

test("initModel total parameter count", () => {
  initModel(100);
  const np = numParams();
  // wte: 100*64 = 6400
  // wpe: 256*64 = 16384
  // per layer: 4*(64*64) + (256*64) + (64*256) = 4*4096 + 16384 + 16384 = 49152
  // 2 layers: 98304
  // total: 6400 + 16384 + 98304 = 121088
  expect(np).equal(121088);
});

test("params are in sorted key order", () => {
  initModel(100);
  // First should be layer0.attn_wk, last should be wte
  const keys = stateDict.keys();
  keys.sort((a: string, b: string): i32 => {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
  });
  // Verify params matches sorted order
  for (let i: i32 = 0; i < keys.length; i++) {
    const fromDict = stateDict.get(keys[i]);
    const fromParams = params[i];
    expect(changetype<usize>(fromDict)).equal(changetype<usize>(fromParams));
  }
});

// ===== Forward pass =====

test("gpt returns logits of correct shape", () => {
  initModel(50);
  const cacheKeys = new Array<Array<Tensor>>(N_LAYER);
  const cacheVals = new Array<Array<Tensor>>(N_LAYER);
  for (let li: i32 = 0; li < N_LAYER; li++) {
    cacheKeys[li] = new Array<Tensor>();
    cacheVals[li] = new Array<Tensor>();
  }
  const logits = gpt(0, 0, cacheKeys, cacheVals);
  expect(logits.data.length).equal(50);
  expect(logits.shape[0]).equal(50);
});

test("gpt logits are finite", () => {
  initModel(50);
  const cacheKeys = new Array<Array<Tensor>>(N_LAYER);
  const cacheVals = new Array<Array<Tensor>>(N_LAYER);
  for (let li: i32 = 0; li < N_LAYER; li++) {
    cacheKeys[li] = new Array<Tensor>();
    cacheVals[li] = new Array<Tensor>();
  }
  const logits = gpt(0, 0, cacheKeys, cacheVals);
  for (let i: i32 = 0; i < logits.data.length; i++) {
    expect(isFinite(logits.data[i])).equal(true);
  }
});

test("gpt different tokens produce different logits", () => {
  initModel(50);
  // Token 0
  const ck1 = new Array<Array<Tensor>>(N_LAYER);
  const cv1 = new Array<Array<Tensor>>(N_LAYER);
  for (let li: i32 = 0; li < N_LAYER; li++) {
    ck1[li] = new Array<Tensor>();
    cv1[li] = new Array<Tensor>();
  }
  const l1 = gpt(0, 0, ck1, cv1);

  // Token 1
  const ck2 = new Array<Array<Tensor>>(N_LAYER);
  const cv2 = new Array<Array<Tensor>>(N_LAYER);
  for (let li: i32 = 0; li < N_LAYER; li++) {
    ck2[li] = new Array<Tensor>();
    cv2[li] = new Array<Tensor>();
  }
  const l2 = gpt(1, 0, ck2, cv2);

  // At least some logits should differ
  let anyDiff = false;
  for (let i: i32 = 0; i < l1.data.length; i++) {
    if (l1.data[i] != l2.data[i]) {
      anyDiff = true;
      break;
    }
  }
  expect(anyDiff).equal(true);
});

test("gpt KV cache grows with sequence", () => {
  initModel(50);
  const cacheKeys = new Array<Array<Tensor>>(N_LAYER);
  const cacheVals = new Array<Array<Tensor>>(N_LAYER);
  for (let li: i32 = 0; li < N_LAYER; li++) {
    cacheKeys[li] = new Array<Tensor>();
    cacheVals[li] = new Array<Tensor>();
  }
  gpt(0, 0, cacheKeys, cacheVals);
  expect(cacheKeys[0].length).equal(1);
  expect(cacheVals[0].length).equal(1);
  gpt(1, 1, cacheKeys, cacheVals);
  expect(cacheKeys[0].length).equal(2);
  expect(cacheVals[0].length).equal(2);
});

// ===== weightedChoice =====

test("weightedChoice returns valid index", () => {
  const probs = new StaticArray<f32>(5);
  for (let i: i32 = 0; i < 5; i++) {
    probs[i] = f32(0.2);
  }
  for (let trial: i32 = 0; trial < 50; trial++) {
    const idx = weightedChoice(probs);
    expect(idx >= 0).equal(true);
    expect(idx < 5).equal(true);
  }
});

test("weightedChoice selects one-hot correctly", () => {
  const probs = new StaticArray<f32>(4);
  probs[0] = f32(0.0);
  probs[1] = f32(0.0);
  probs[2] = f32(1.0);
  probs[3] = f32(0.0);
  for (let trial: i32 = 0; trial < 20; trial++) {
    expect(weightedChoice(probs)).equal(2);
  }
});

// ===== detachKvCache =====

test("detachKvCache clears children on cached tensors", () => {
  initModel(50);
  const cacheKeys = new Array<Array<Tensor>>(N_LAYER);
  const cacheVals = new Array<Array<Tensor>>(N_LAYER);
  for (let li: i32 = 0; li < N_LAYER; li++) {
    cacheKeys[li] = new Array<Tensor>();
    cacheVals[li] = new Array<Tensor>();
  }
  gpt(0, 0, cacheKeys, cacheVals);
  // Before detach, cached tensors have children from computation graph
  expect(cacheKeys[0][0].children.length > 0).equal(true);
  detachKvCache(cacheKeys, cacheVals);
  // After detach, all children cleared
  for (let li: i32 = 0; li < N_LAYER; li++) {
    for (let ti: i32 = 0; ti < cacheKeys[li].length; ti++) {
      expect(cacheKeys[li][ti].children.length).equal(0);
      expect(cacheVals[li][ti].children.length).equal(0);
    }
  }
});
