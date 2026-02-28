// checkpoint.ts — Binary checkpoint persistence
// Depends on model.ts for state dict access

import { FileSystem, Descriptor } from "as-wasi/assembly";
import {
  fd_write, fd_read, errno
} from "@assemblyscript/wasi-shim/assembly/bindings/wasi_snapshot_preview1";
import {
  stateDict, params, vocabSize,
  N_EMBD, N_LAYER, N_HEAD, BLOCK_SIZE,
} from "./model";

// ===== Low-level binary I/O helpers =====

function writeRawBytes(rawfd: u32, ptr: usize, len: i32): bool {
  const iov = memory.data(16);
  store<u32>(iov, u32(ptr), 0);
  store<u32>(iov, u32(len), 4);
  const writtenPtr = memory.data(8);
  return fd_write(rawfd, iov, 1, writtenPtr) == errno.SUCCESS;
}

function readRawBytes(rawfd: u32, ptr: usize, len: i32): bool {
  let remaining = len;
  let offset: usize = 0;
  while (remaining > 0) {
    const iov = memory.data(16);
    store<u32>(iov, u32(ptr + offset), 0);
    store<u32>(iov, u32(remaining), 4);
    const readPtr = memory.data(8);
    if (fd_read(rawfd, iov, 1, readPtr) != errno.SUCCESS) {
      return false;
    }
    const bytesRead = load<u32>(readPtr);
    if (bytesRead == 0) return false; // EOF
    offset += usize(bytesRead);
    remaining -= i32(bytesRead);
  }
  return true;
}

function writeI32(rawfd: u32, value: i32): bool {
  const buf = memory.data(4);
  store<i32>(buf, value);
  return writeRawBytes(rawfd, buf, 4);
}

function readI32(rawfd: u32): i32 {
  const buf = memory.data(4);
  readRawBytes(rawfd, buf, 4);
  return load<i32>(buf);
}

function writeStaticArrayF32(rawfd: u32, arr: StaticArray<f32>): bool {
  return writeRawBytes(rawfd, changetype<usize>(arr), arr.length << 2);
}

function readStaticArrayF32(rawfd: u32, arr: StaticArray<f32>): bool {
  return readRawBytes(rawfd, changetype<usize>(arr), arr.length << 2);
}

// ===== Sort helper =====

function stringLessThan(a: string, b: string): i32 {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

// ===== Checkpoint save =====

export function saveCheckpoint(
  path: string,
  step: i32,
  adamM: Map<string, StaticArray<f32>>,
  adamV: Map<string, StaticArray<f32>>
): void {
  const fd = FileSystem.open(path, "w");
  if (fd === null) {
    abort();
  }
  const rawfd = (fd as Descriptor).rawfd;

  // Header
  writeI32(rawfd, step);
  writeI32(rawfd, N_EMBD);
  writeI32(rawfd, N_LAYER);
  writeI32(rawfd, N_HEAD);
  writeI32(rawfd, BLOCK_SIZE);
  writeI32(rawfd, vocabSize);

  // Weights in sorted key order
  const keys = stateDict.keys();
  keys.sort(stringLessThan);
  for (let i: i32 = 0; i < keys.length; i++) {
    const t = stateDict.get(keys[i]);
    writeStaticArrayF32(rawfd, t.data);
  }

  // Adam M in sorted key order
  for (let i: i32 = 0; i < keys.length; i++) {
    writeStaticArrayF32(rawfd, adamM.get(keys[i]));
  }

  // Adam V in sorted key order
  for (let i: i32 = 0; i < keys.length; i++) {
    writeStaticArrayF32(rawfd, adamV.get(keys[i]));
  }
}

// ===== Checkpoint load =====

export function loadCheckpoint(
  path: string,
  adamM: Map<string, StaticArray<f32>>,
  adamV: Map<string, StaticArray<f32>>
): i32 {
  const fd = FileSystem.open(path, "r");
  if (fd === null) {
    return -1; // No checkpoint
  }
  const rawfd = (fd as Descriptor).rawfd;

  // Read and validate header
  const step = readI32(rawfd);
  const savedEmbd = readI32(rawfd);
  const savedLayer = readI32(rawfd);
  const savedHead = readI32(rawfd);
  const savedBlock = readI32(rawfd);
  const savedVocab = readI32(rawfd);

  if (savedEmbd != N_EMBD || savedLayer != N_LAYER ||
      savedHead != N_HEAD || savedBlock != BLOCK_SIZE ||
      savedVocab != vocabSize) {
    return -2; // Config mismatch
  }

  // Read weights in sorted key order
  const keys = stateDict.keys();
  keys.sort(stringLessThan);
  for (let i: i32 = 0; i < keys.length; i++) {
    const t = stateDict.get(keys[i]);
    readStaticArrayF32(rawfd, t.data);
  }

  // Read Adam M
  for (let i: i32 = 0; i < keys.length; i++) {
    readStaticArrayF32(rawfd, adamM.get(keys[i]));
  }

  // Read Adam V
  for (let i: i32 = 0; i < keys.length; i++) {
    readStaticArrayF32(rawfd, adamV.get(keys[i]));
  }

  return step;
}
