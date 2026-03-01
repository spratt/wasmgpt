// io.ts — Safe file reading for AssemblyScript/WASI.
// Descriptor.readString() and readAll() use memory.data() for iov/read_ptr
// buffers, which are static addresses that can be clobbered by other code.
// This module uses heap-allocated buffers to avoid the corruption.

import { Descriptor } from "as-wasi/assembly";
import { fd_read, errno } from "@assemblyscript/wasi-shim/assembly/bindings/wasi_snapshot_preview1";

export function readFileText(desc: Descriptor): string | null {
  const rawfd = desc.rawfd;
  const chunkSize: usize = 4096;
  const buf = new ArrayBuffer(chunkSize as i32);
  const bufPtr = changetype<usize>(buf);

  // Heap-allocate iov and read_ptr to avoid memory.data() conflicts
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
