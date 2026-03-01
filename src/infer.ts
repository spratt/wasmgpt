// infer.ts — Generate WAT text from a trained GPT-2 model
// WASI CLI entry point.
//
// Usage:
//   wasmtime --dir . build/infer.wasm <vocab.sexp> [numSamples] [temperature] [prompt...]

import { CommandLine, Console, FileSystem, Descriptor } from "as-wasi/assembly";
import { readFileText } from "./io";
import { tokenize } from "./lexer";
import { parseVocab } from "./bpe";
import {
  initModel, gpt, stateDict, weightedChoice, detachKvCache,
  setHyperparams, getNEmbd, getNLayer, getNHead, getBlockSize,
} from "./model";
import { loadCheckpoint } from "./checkpoint";
import { Tensor, softmax, divScalar } from "./tensor";
import { parseConfig, configI32, configF32 } from "./config";

// ===== Parse CLI args =====

const args = CommandLine.all;

if (args.length < 2) {
  Console.error("Usage: infer <vocab.sexp> [numSamples] [temperature] [prompt...]\n");
  abort();
}

const vocabPath = args[1];
let numSamples: i32 = 5;
if (args.length >= 3) {
  numSamples = I32.parseInt(args[2]);
}
let cliTemperature: f32 = f32(-1.0);
if (args.length >= 4) {
  cliTemperature = f32(parseFloat(args[3]));
}
let promptText = "";
if (args.length >= 5) {
  const parts = new Array<string>();
  for (let i: i32 = 4; i < args.length; i++) {
    parts.push(args[i]);
  }
  promptText = parts.join(" ");
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

const temperature: f32 = cliTemperature > f32(0.0)
  ? cliTemperature
  : configF32(config, "temperature", f32(0.8));

// ===== Load vocabulary from S-expression =====

const vocabFd = FileSystem.open(vocabPath, "r");
if (vocabFd === null) {
  Console.error("Error: could not open vocab file: " + vocabPath + "\n");
  abort();
}
const vocabText = readFileText(vocabFd as Descriptor);
if (vocabText === null) {
  Console.error("Error: could not read vocab file: " + vocabPath + "\n");
  abort();
}

const tokenToId = parseVocab(vocabText as string);
const idToToken = new Map<i32, string>();
let totalVocab: i32 = 0;

const vocabKeys = tokenToId.keys();
for (let i: i32 = 0; i < vocabKeys.length; i++) {
  const token = vocabKeys[i];
  const id = tokenToId.get(token);
  idToToken.set(id, token);
  if (id >= totalVocab) totalVocab = id + 1;
}

Console.error("vocabulary: " + totalVocab.toString() + " tokens\n");

// ===== BOS token =====

const BOS_TOKEN = "<BOS>";
if (!tokenToId.has(BOS_TOKEN)) {
  Console.error("ERROR: no <BOS> token in vocabulary\n");
  abort();
}
const BOS_ID: i32 = tokenToId.get(BOS_TOKEN);

// ===== Sort helper =====

function stringLessThan(a: string, b: string): i32 {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

// ===== Initialize model and load checkpoint =====

initModel(totalVocab);

const CHECKPOINT_PATH: string = "build/model.bin";

// Dummy Adam maps for loadCheckpoint
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

const loaded = loadCheckpoint(CHECKPOINT_PATH, adamM, adamV);
if (loaded >= 0) {
  Console.error("loaded checkpoint: step " + loaded.toString() + "\n");
} else {
  Console.error("ERROR: failed to load checkpoint (code " + loaded.toString() + ")\n");
  abort();
}

// ===== Encode prompt =====

function encodePrompt(text: string): Array<i32> {
  const tokens = tokenize(text);
  const ids = new Array<i32>();
  for (let i: i32 = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (tokenToId.has(tok)) {
      ids.push(tokenToId.get(tok));
    } else {
      Console.error("warning: unknown prompt token: " + tok + "\n");
    }
  }
  return ids;
}

// ===== Decode token IDs to text =====

function decodeIds(ids: Array<i32>): string {
  let result = "";
  for (let i: i32 = 0; i < ids.length; i++) {
    const tok = idToToken.has(ids[i]) ? idToToken.get(ids[i]) : "<?>";
    if (i > 0) {
      const prev = idToToken.has(ids[i - 1]) ? idToToken.get(ids[i - 1]) : "<?>";
      if (prev != "(" && tok != ")") {
        result += " ";
      }
    }
    result += tok;
  }
  return result;
}

// ===== Inference =====

Console.error("inference: " + numSamples.toString() + " samples, temperature " + temperature.toString() + "\n");

for (let s: i32 = 0; s < numSamples; s++) {
  const nLayerVal = getNLayer();
  const blockSizeVal = getBlockSize();
  const cacheKeys = new Array<Array<Tensor>>(nLayerVal);
  const cacheVals = new Array<Array<Tensor>>(nLayerVal);
  for (let li: i32 = 0; li < nLayerVal; li++) {
    cacheKeys[li] = new Array<Tensor>();
    cacheVals[li] = new Array<Tensor>();
  }

  const generated = new Array<i32>();
  let tokenId: i32 = BOS_ID;
  let posId: i32 = 0;

  // Feed BOS through model at position 0
  gpt(tokenId, posId, cacheKeys, cacheVals);
  detachKvCache(cacheKeys, cacheVals);
  posId++;

  // Feed prompt through model
  if (promptText.length > 0) {
    const promptIds = encodePrompt(promptText);
    for (let i: i32 = 0; i < promptIds.length; i++) {
      tokenId = promptIds[i];
      generated.push(tokenId);
      gpt(tokenId, posId, cacheKeys, cacheVals);
      detachKvCache(cacheKeys, cacheVals);
      posId++;
    }
  }

  // Autoregressive generation
  for (let p: i32 = posId; p < blockSizeVal; p++) {
    const logits = gpt(tokenId, p, cacheKeys, cacheVals);
    const scaled = divScalar(logits, temperature);
    const probs = softmax(scaled);
    const nextId = weightedChoice(probs.data);
    detachKvCache(cacheKeys, cacheVals);
    if (nextId == BOS_ID) break;
    generated.push(nextId);
    tokenId = nextId;
  }

  const text = decodeIds(generated);
  Console.write("sample " + (s + 1).toString() + ": " + text + "\n", false);
}
