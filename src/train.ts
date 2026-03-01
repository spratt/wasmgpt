// train.ts — Train a GPT-2 model on WAT token IDs
// WASI CLI entry point.
//
// Usage:
//   wasmtime --dir . build/train.wasm <corpus.wat> <merges.sexp> [numSteps]

import { CommandLine, Console, FileSystem, Descriptor } from "as-wasi/assembly";
import { readFileText } from "./io";
import { tokenize } from "./lexer";
import { vocab, getNextId, getBosId, initVocabulary } from "./vocabulary";
import { buildBpeVocab, bpeEncodeToken, parseMerges, getBpeNextId, tokenToChars } from "./bpe";
import {
  initModel, gpt, stateDict, params, vocabSize,
  setHyperparams, getNEmbd, getNLayer, getNHead, getBlockSize,
} from "./model";
import { saveCheckpoint, loadCheckpoint } from "./checkpoint";
import { Tensor, backward, crossEntropy, divScalar, tensorSum, concat } from "./tensor";
import { parseConfig, configI32, configF32 } from "./config";

// ===== Parse CLI args =====

const args = CommandLine.all;

if (args.length < 3) {
  Console.error("Usage: train <corpus.wat> <merges.sexp> [numSteps]\n");
  abort();
}

const corpusPath = args[1];
const mergesPath = args[2];
let numSteps: i32 = 100;
if (args.length >= 4) {
  numSteps = I32.parseInt(args[3]);
}

// ===== Load configuration =====

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

const nEmbd = configI32(config, "n-embd", getNEmbd());
const nLayer = configI32(config, "n-layer", getNLayer());
const nHead = configI32(config, "n-head", getNHead());
const blockSize = configI32(config, "block-size", getBlockSize());
const initScale = configF32(config, "init-scale", f32(0.02));
setHyperparams(nEmbd, nLayer, nHead, blockSize, initScale);

const TRAIN_SEQ_LEN: i32 = configI32(config, "train-seq-len", 32);
const LEARNING_RATE: f32 = configF32(config, "learning-rate", f32(0.001));
const BETA1: f32 = configF32(config, "beta1", f32(0.9));
const BETA2: f32 = configF32(config, "beta2", f32(0.999));
const EPS_ADAM: f32 = configF32(config, "eps-adam", f32(1e-8));
const CHECKPOINT_INTERVAL: i32 = configI32(config, "checkpoint-interval", 10);
const CHECKPOINT_PATH: string = "build/model.bin";

Console.error("config: n-embd=" + nEmbd.toString() + " n-layer=" + nLayer.toString()
  + " n-head=" + nHead.toString() + " block-size=" + blockSize.toString()
  + " train-seq-len=" + TRAIN_SEQ_LEN.toString() + "\n");

// ===== Sort helper =====

function stringLessThan(a: string, b: string): i32 {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

// ===== Initialize tokenizer =====

initVocabulary();

// Load BPE merges
const mergesFd = FileSystem.open(mergesPath, "r");
if (mergesFd === null) {
  Console.error("Error: could not open merges file: " + mergesPath + "\n");
  abort();
}
const mergesText = readFileText(mergesFd as Descriptor);
if (mergesText === null) {
  Console.error("Error: could not read merges file: " + mergesPath + "\n");
  abort();
}
const merges = parseMerges(mergesText as string);
const pass1Size = getNextId();

// ===== Read and tokenize corpus =====

const corpusFd = FileSystem.open(corpusPath, "r");
if (corpusFd === null) {
  Console.error("Error: could not open corpus: " + corpusPath + "\n");
  abort();
}
const corpusText = readFileText(corpusFd as Descriptor);
if (corpusText === null) {
  Console.error("Error: could not read corpus: " + corpusPath + "\n");
  abort();
}

const tokens = tokenize(corpusText as string);
Console.error("corpus: " + tokens.length.toString() + " tokens\n");

// Collect all unique characters from unknown tokens so every character
// gets a BPE ID, even if it never participated in a merge rule.
const corpusChars = new Array<string>();
const charSeen = new Map<string, bool>();
for (let i: i32 = 0; i < tokens.length; i++) {
  if (!vocab.has(tokens[i])) {
    const chars = tokenToChars(tokens[i]);
    for (let j: i32 = 0; j < chars.length; j++) {
      if (!charSeen.has(chars[j])) {
        charSeen.set(chars[j], true);
        corpusChars.push(chars[j]);
      }
    }
  }
}

buildBpeVocab(merges, pass1Size, corpusChars);
Console.error("vocabulary: " + pass1Size.toString() + " pass1, " + getBpeNextId().toString() + " total\n");

// Encode tokens to IDs
const allIds = new Array<i32>();
for (let i: i32 = 0; i < tokens.length; i++) {
  const tok = tokens[i];
  if (vocab.has(tok)) {
    allIds.push(vocab.get(tok));
  } else {
    const bpeIds = bpeEncodeToken(tok);
    for (let j: i32 = 0; j < bpeIds.length; j++) {
      allIds.push(bpeIds[j]);
    }
  }
}
// Wrap corpus with BOS (following microgpt.py)
const bosId = getBosId();
allIds.unshift(bosId);
allIds.push(bosId);
Console.error("encoded: " + allIds.length.toString() + " token IDs (including BOS)\n");

// ===== Initialize model =====

const totalVocab = getBpeNextId();
initModel(totalVocab);
Console.error("model: " + params.length.toString() + " tensors\n");

// ===== Adam optimizer state =====

const adamM = new Map<string, StaticArray<f32>>();
const adamV = new Map<string, StaticArray<f32>>();
const keys = stateDict.keys();
keys.sort(stringLessThan);
for (let i: i32 = 0; i < keys.length; i++) {
  const key = keys[i];
  const t = stateDict.get(key);
  adamM.set(key, new StaticArray<f32>(t.data.length));
  adamV.set(key, new StaticArray<f32>(t.data.length));
}

// ===== Load checkpoint if available =====

let startStep: i32 = 0;
const loaded = loadCheckpoint(CHECKPOINT_PATH, adamM, adamV);
if (loaded >= 0) {
  startStep = loaded;
  Console.error("resumed from step " + startStep.toString() + "\n");
} else if (loaded < -1) {
  Console.error("ERROR: failed to load checkpoint (code " + loaded.toString() + ")\n");
  abort();
}

// ===== Batch extraction =====

function getBatch(step: i32): StaticArray<i32> {
  const n = allIds.length;
  const maxStart = n - TRAIN_SEQ_LEN - 1;
  const start = maxStart > 0 ? (step * TRAIN_SEQ_LEN) % maxStart : 0;
  const batch = new StaticArray<i32>(TRAIN_SEQ_LEN + 1);
  for (let i: i32 = 0; i < TRAIN_SEQ_LEN + 1; i++) {
    batch[i] = allIds[start + i];
  }
  return batch;
}

// ===== Training loop =====

const totalSteps = startStep + numSteps;
Console.error("training: " + numSteps.toString() + " steps (" + (startStep + 1).toString() + " to " + totalSteps.toString() + ")\n");

for (let i: i32 = 0; i < numSteps; i++) {
  const step = startStep + i;
  const batch = getBatch(step);

  // Fresh KV cache
  const nLayerVal = getNLayer();
  const cacheKeys = new Array<Array<Tensor>>(nLayerVal);
  const cacheVals = new Array<Array<Tensor>>(nLayerVal);
  for (let li: i32 = 0; li < nLayerVal; li++) {
    cacheKeys[li] = new Array<Tensor>();
    cacheVals[li] = new Array<Tensor>();
  }

  // Forward pass: autoregressive over sequence
  const losses = new Array<Tensor>(TRAIN_SEQ_LEN);
  for (let posId: i32 = 0; posId < TRAIN_SEQ_LEN; posId++) {
    const tokenId = batch[posId];
    const targetId = batch[posId + 1];
    const logits = gpt(tokenId, posId, cacheKeys, cacheVals);
    losses[posId] = crossEntropy(logits, targetId);
  }

  // Average loss
  const loss = divScalar(tensorSum(concat(losses)), f32(TRAIN_SEQ_LEN));

  // Backward
  backward(loss);

  // Adam update
  const lrT = LEARNING_RATE * (f32(1.0) - f32(i) / f32(numSteps));
  const stepF = f32(step + 1);

  for (let ki: i32 = 0; ki < keys.length; ki++) {
    const key = keys[ki];
    const p = stateDict.get(key);
    const m = adamM.get(key);
    const v = adamV.get(key);
    const bc1 = f32(1.0) - Mathf.pow(BETA1, stepF);
    const bc2 = f32(1.0) - Mathf.pow(BETA2, stepF);

    for (let j: i32 = 0; j < p.data.length; j++) {
      const g = p.grad[j];
      m[j] = BETA1 * m[j] + (f32(1.0) - BETA1) * g;
      v[j] = BETA2 * v[j] + (f32(1.0) - BETA2) * g * g;
      const mHat = m[j] / bc1;
      const vHat = v[j] / bc2;
      p.data[j] -= lrT * mHat / (Mathf.sqrt(vHat) + EPS_ADAM);
      p.grad[j] = f32(0.0);
    }
  }

  Console.error("step " + (step + 1).toString() + " / " + totalSteps.toString() + " | loss " + loss.data[0].toString() + "\n");

  if ((i + 1) % CHECKPOINT_INTERVAL == 0) {
    saveCheckpoint(CHECKPOINT_PATH, step + 1, adamM, adamV);
    Console.error("checkpoint saved: step " + (step + 1).toString() + "\n");
  }
}

// Final checkpoint if not aligned
if (numSteps % CHECKPOINT_INTERVAL != 0) {
  saveCheckpoint(CHECKPOINT_PATH, startStep + numSteps, adamM, adamV);
  Console.error("checkpoint saved: step " + (startStep + numSteps).toString() + "\n");
}
