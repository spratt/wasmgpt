// bench-io.ts — Test harness for file I/O strategies.
// Reads a file using different approaches and reports results.
// Does NOT modify production code.
//
// Usage:
//   wasmtime --dir . build/bench-io.wasm <file>

import { CommandLine, Console, FileSystem, Descriptor } from "as-wasi/assembly";
import { fd_read, errno } from "@assemblyscript/wasi-shim/assembly/bindings/wasi_snapshot_preview1";

// ===== Strategy 1: Current production code (char-by-char concat) =====

function readCharByChar(desc: Descriptor): string | null {
  const rawfd = desc.rawfd;
  const chunkSize: usize = 4096;
  const buf = new ArrayBuffer(chunkSize as i32);
  const bufPtr = changetype<usize>(buf);

  const iov = new ArrayBuffer(8);
  const iovPtr = changetype<usize>(iov);
  const readBuf = new ArrayBuffer(4);
  const readPtr = changetype<usize>(readBuf);

  let result = "";

  while (true) {
    store<u32>(iovPtr, bufPtr, 0);
    store<u32>(iovPtr, chunkSize as u32, 4);

    if (fd_read(rawfd, iovPtr, 1, readPtr) !== errno.SUCCESS) {
      return null;
    }

    const bytesRead = load<u32>(readPtr);
    if (bytesRead == 0) break;

    for (let i: u32 = 0; i < bytesRead; i++) {
      result += String.fromCharCode(load<u8>(bufPtr + i) as i32);
    }
  }

  return result;
}

// ===== Strategy 2: String.UTF8.decodeUnsafe per chunk, join at end =====

function readDecodeUnsafe(desc: Descriptor): string | null {
  const rawfd = desc.rawfd;
  const chunkSize: usize = 4096;
  const buf = new ArrayBuffer(chunkSize as i32);
  const bufPtr = changetype<usize>(buf);

  const iov = new ArrayBuffer(8);
  const iovPtr = changetype<usize>(iov);
  const readBuf = new ArrayBuffer(4);
  const readPtr = changetype<usize>(readBuf);

  const chunks = new Array<string>();

  while (true) {
    store<u32>(iovPtr, bufPtr, 0);
    store<u32>(iovPtr, chunkSize as u32, 4);

    if (fd_read(rawfd, iovPtr, 1, readPtr) !== errno.SUCCESS) {
      return null;
    }

    const bytesRead = load<u32>(readPtr);
    if (bytesRead == 0) break;

    chunks.push(String.UTF8.decodeUnsafe(bufPtr, bytesRead, false));
  }

  return chunks.join("");
}

// ===== Strategy 3: Collect all bytes, decode once =====

function readDecodeOnce(desc: Descriptor): string | null {
  const rawfd = desc.rawfd;
  const chunkSize: usize = 4096;
  const buf = new ArrayBuffer(chunkSize as i32);
  const bufPtr = changetype<usize>(buf);

  const iov = new ArrayBuffer(8);
  const iovPtr = changetype<usize>(iov);
  const readBuf = new ArrayBuffer(4);
  const readPtr = changetype<usize>(readBuf);

  // Collect raw bytes into a growable buffer
  let totalBytes: i32 = 0;
  let capacity: i32 = 65536;
  let collected = new ArrayBuffer(capacity);
  let collectedPtr = changetype<usize>(collected);

  while (true) {
    store<u32>(iovPtr, bufPtr, 0);
    store<u32>(iovPtr, chunkSize as u32, 4);

    if (fd_read(rawfd, iovPtr, 1, readPtr) !== errno.SUCCESS) {
      return null;
    }

    const bytesRead = load<u32>(readPtr);
    if (bytesRead == 0) break;

    // Grow if needed
    if (totalBytes + (bytesRead as i32) > capacity) {
      const newCapacity = capacity * 2;
      const newBuf = new ArrayBuffer(newCapacity);
      const newPtr = changetype<usize>(newBuf);
      memory.copy(newPtr, collectedPtr, totalBytes as usize);
      collected = newBuf;
      collectedPtr = newPtr;
      capacity = newCapacity;
    }

    memory.copy(collectedPtr + (totalBytes as usize), bufPtr, bytesRead);
    totalBytes += bytesRead as i32;
  }

  return String.UTF8.decodeUnsafe(collectedPtr, totalBytes as usize, false);
}

// ===== Strategy 4: String.UTF8.decode (safe wrapper) per chunk =====

function readDecodeSafe(desc: Descriptor): string | null {
  const rawfd = desc.rawfd;
  const chunkSize: usize = 4096;
  const buf = new ArrayBuffer(chunkSize as i32);
  const bufPtr = changetype<usize>(buf);

  const iov = new ArrayBuffer(8);
  const iovPtr = changetype<usize>(iov);
  const readBuf = new ArrayBuffer(4);
  const readPtr = changetype<usize>(readBuf);

  const chunks = new Array<string>();

  while (true) {
    store<u32>(iovPtr, bufPtr, 0);
    store<u32>(iovPtr, chunkSize as u32, 4);

    if (fd_read(rawfd, iovPtr, 1, readPtr) !== errno.SUCCESS) {
      return null;
    }

    const bytesRead = load<u32>(readPtr);
    if (bytesRead == 0) break;

    // Copy only the bytes read into a right-sized ArrayBuffer
    const slice = new ArrayBuffer(bytesRead as i32);
    memory.copy(changetype<usize>(slice), bufPtr, bytesRead);
    chunks.push(String.UTF8.decode(slice, false));
  }

  return chunks.join("");
}

// ===== Main =====

const args = CommandLine.all;

if (args.length < 2) {
  Console.error("Usage: bench-io <file>\n");
  abort();
}

const filePath = args[1];

// Strategy 1: char-by-char
let fd = FileSystem.open(filePath, "r");
if (fd === null) {
  Console.error("Error: could not open file: " + filePath + "\n");
  abort();
}
const s1 = readCharByChar(fd as Descriptor);
if (s1 === null) {
  Console.error("Error: readCharByChar failed\n");
  abort();
}
Console.error("strategy 1 (char-by-char):     " + (s1 as string).length.toString() + " chars\n");

// Strategy 2: decodeUnsafe per chunk
fd = FileSystem.open(filePath, "r");
if (fd === null) {
  Console.error("Error: could not open file\n");
  abort();
}
const s2 = readDecodeUnsafe(fd as Descriptor);
if (s2 === null) {
  Console.error("Error: readDecodeUnsafe failed\n");
  abort();
}
Console.error("strategy 2 (decodeUnsafe):     " + (s2 as string).length.toString() + " chars\n");

// Strategy 3: collect bytes, decode once
fd = FileSystem.open(filePath, "r");
if (fd === null) {
  Console.error("Error: could not open file\n");
  abort();
}
const s3 = readDecodeOnce(fd as Descriptor);
if (s3 === null) {
  Console.error("Error: readDecodeOnce failed\n");
  abort();
}
Console.error("strategy 3 (decode once):      " + (s3 as string).length.toString() + " chars\n");

// Strategy 4: decode safe per chunk
fd = FileSystem.open(filePath, "r");
if (fd === null) {
  Console.error("Error: could not open file\n");
  abort();
}
const s4 = readDecodeSafe(fd as Descriptor);
if (s4 === null) {
  Console.error("Error: readDecodeSafe failed\n");
  abort();
}
Console.error("strategy 4 (decode safe):      " + (s4 as string).length.toString() + " chars\n");

// Verify all strategies produce the same result
const match12 = (s1 as string) == (s2 as string);
const match13 = (s1 as string) == (s3 as string);
const match14 = (s1 as string) == (s4 as string);
Console.error("\nmatch 1==2: " + match12.toString() + "\n");
Console.error("match 1==3: " + match13.toString() + "\n");
Console.error("match 1==4: " + match14.toString() + "\n");

if (!match12 || !match13 || !match14) {
  // Show where they diverge
  const ref = s1 as string;
  const candidates: Array<string> = [s2 as string, s3 as string, s4 as string];
  const names: Array<string> = ["s2", "s3", "s4"];
  for (let c: i32 = 0; c < candidates.length; c++) {
    const other = candidates[c];
    if (ref != other) {
      const minLen = ref.length < other.length ? ref.length : other.length;
      for (let i: i32 = 0; i < minLen; i++) {
        if (ref.charCodeAt(i) != other.charCodeAt(i)) {
          Console.error("first diff " + names[c] + " at char " + i.toString()
            + ": ref=" + ref.charCodeAt(i).toString()
            + " " + names[c] + "=" + other.charCodeAt(i).toString() + "\n");
          break;
        }
      }
      if (ref.length != other.length) {
        Console.error("length diff " + names[c] + ": ref=" + ref.length.toString()
          + " " + names[c] + "=" + other.length.toString() + "\n");
      }
    }
  }
}
