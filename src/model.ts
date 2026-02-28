// model.ts — GPT-2 architecture (Tiny config)
// Depends on tensor.ts

import {
  Tensor, tensorFrom, tensorZeros,
  add, relu, matmul, softmax, rmsnorm, embedding,
  slice, concat, mul, tensorSum, divScalar, scale,
} from "./tensor";

// ===== PRNG (LCG, same as consgpt.lisp) =====

export let lcgState: u32 = 42;

export function lcgNext(): u32 {
  lcgState = u32(u64(1664525) * u64(lcgState) + u64(1013904223)) & 0xFFFFFFFF;
  return lcgState;
}

export function randomUniform(): f32 {
  return f32(lcgNext()) / f32(4294967296.0);
}

export function randomGaussian(): f32 {
  const u1 = Mathf.max(randomUniform(), f32(1e-10));
  const u2 = randomUniform();
  return Mathf.sqrt(f32(-2.0) * Mathf.log(u1)) * Mathf.cos(f32(2.0) * f32(3.141592653589793) * u2);
}

// ===== Hyperparameters =====

export const N_EMBD: i32 = 64;
export const N_LAYER: i32 = 2;
export const N_HEAD: i32 = 4;
export const HEAD_DIM: i32 = 16; // N_EMBD / N_HEAD
export const BLOCK_SIZE: i32 = 256;

// ===== Weight initialization =====

export function makeMatrix(nout: i32, nin: i32): Tensor {
  const data = new StaticArray<f32>(nout * nin);
  for (let i: i32 = 0; i < nout * nin; i++) {
    data[i] = randomGaussian() * f32(0.02);
  }
  const shape = new StaticArray<i32>(2);
  shape[0] = nout;
  shape[1] = nin;
  return tensorFrom(data, shape);
}

// ===== State dictionary =====

export let stateDict: Map<string, Tensor> = new Map<string, Tensor>();
export let params: Array<Tensor> = new Array<Tensor>(0);
export let vocabSize: i32 = 0;

function stringLessThan(a: string, b: string): i32 {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

export function initModel(vs: i32): void {
  vocabSize = vs;
  stateDict = new Map<string, Tensor>();

  stateDict.set("wte", makeMatrix(vocabSize, N_EMBD));
  stateDict.set("wpe", makeMatrix(BLOCK_SIZE, N_EMBD));

  for (let li: i32 = 0; li < N_LAYER; li++) {
    const prefix = "layer" + li.toString() + ".";
    stateDict.set(prefix + "attn_wq", makeMatrix(N_EMBD, N_EMBD));
    stateDict.set(prefix + "attn_wk", makeMatrix(N_EMBD, N_EMBD));
    stateDict.set(prefix + "attn_wv", makeMatrix(N_EMBD, N_EMBD));
    stateDict.set(prefix + "attn_wo", makeMatrix(N_EMBD, N_EMBD));
    stateDict.set(prefix + "mlp_fc1", makeMatrix(4 * N_EMBD, N_EMBD));
    stateDict.set(prefix + "mlp_fc2", makeMatrix(N_EMBD, 4 * N_EMBD));
  }

  // Collect all tensors in sorted key order for optimizer
  const keys = stateDict.keys();
  keys.sort(stringLessThan);
  params = new Array<Tensor>(keys.length);
  for (let i: i32 = 0; i < keys.length; i++) {
    params[i] = stateDict.get(keys[i]);
  }
}

export function numParams(): i32 {
  let total: i32 = 0;
  for (let i: i32 = 0; i < params.length; i++) {
    total += params[i].data.length;
  }
  return total;
}

// ===== Forward pass =====

export function gpt(
  tokenId: i32,
  posId: i32,
  cacheKeys: Array<Array<Tensor>>,
  cacheVals: Array<Array<Tensor>>
): Tensor {
  const wte = stateDict.get("wte");
  const wpe = stateDict.get("wpe");

  // Token + position embedding
  let x = add(embedding(wte, tokenId), embedding(wpe, posId));
  x = rmsnorm(x, f32(1e-5));

  for (let li: i32 = 0; li < N_LAYER; li++) {
    const prefix = "layer" + li.toString() + ".";
    const attnWq = stateDict.get(prefix + "attn_wq");
    const attnWk = stateDict.get(prefix + "attn_wk");
    const attnWv = stateDict.get(prefix + "attn_wv");
    const attnWo = stateDict.get(prefix + "attn_wo");
    const mlpFc1 = stateDict.get(prefix + "mlp_fc1");
    const mlpFc2 = stateDict.get(prefix + "mlp_fc2");

    // Attention block
    const xResidual = x;
    const xn = rmsnorm(x, f32(1e-5));

    const q = matmul(attnWq, xn);
    const k = matmul(attnWk, xn);
    const v = matmul(attnWv, xn);

    // KV cache
    cacheKeys[li].push(k);
    cacheVals[li].push(v);

    // Multi-head attention
    const headOuts = new Array<Tensor>(N_HEAD);
    const seqLen = cacheKeys[li].length;
    const scaleFactor = Mathf.sqrt(f32(HEAD_DIM));

    for (let h: i32 = 0; h < N_HEAD; h++) {
      const hs = h * HEAD_DIM;
      const he = hs + HEAD_DIM;
      const qH = slice(q, hs, he);

      // Compute attention scores
      const scores = new Array<Tensor>(seqLen);
      for (let ti: i32 = 0; ti < seqLen; ti++) {
        const kH = slice(cacheKeys[li][ti], hs, he);
        const dotProd = tensorSum(mul(qH, kH));
        scores[ti] = divScalar(dotProd, scaleFactor);
      }

      const scoreVec = concat(scores);
      const attnWeights = softmax(scoreVec);

      // Weighted sum of values
      let headOut = scale(
        slice(cacheVals[li][0], hs, he),
        slice(attnWeights, 0, 1)
      );
      for (let ti: i32 = 1; ti < seqLen; ti++) {
        headOut = add(headOut, scale(
          slice(cacheVals[li][ti], hs, he),
          slice(attnWeights, ti, ti + 1)
        ));
      }
      headOuts[h] = headOut;
    }

    const xAttn = concat(headOuts);
    x = add(matmul(attnWo, xAttn), xResidual);

    // MLP block
    const xResidual2 = x;
    x = rmsnorm(x, f32(1e-5));
    x = matmul(mlpFc1, x);
    x = relu(x);
    x = matmul(mlpFc2, x);
    x = add(x, xResidual2);
  }

  // Output projection: weight tying with wte
  return matmul(wte, x);
}
