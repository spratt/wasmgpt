// tensor.ts — Tensor computation graph with automatic differentiation
// Replaces scalar autograd (Val per f32) with tensor-based autograd.
// Each weight matrix is a single Tensor backed by contiguous StaticArray<f32>.

// ===== Op enum =====

export const OP_NONE: i32 = 0;
export const OP_MATMUL: i32 = 1;
export const OP_ADD: i32 = 2;
export const OP_RELU: i32 = 3;
export const OP_SOFTMAX: i32 = 4;
export const OP_RMSNORM: i32 = 5;
export const OP_LOG: i32 = 6;
export const OP_NEG: i32 = 7;
export const OP_EMBEDDING: i32 = 8;
export const OP_SUM: i32 = 9;
export const OP_MUL_SCALAR: i32 = 10;
export const OP_DIV_SCALAR: i32 = 11;
export const OP_SLICE: i32 = 12;
export const OP_CONCAT: i32 = 13;
export const OP_MUL: i32 = 14;
export const OP_CROSS_ENTROPY: i32 = 15;
export const OP_SCALE: i32 = 16;

// ===== Tensor class =====

export class Tensor {
  data: StaticArray<f32> = new StaticArray<f32>(0);
  grad: StaticArray<f32> = new StaticArray<f32>(0);
  shape: StaticArray<i32> = new StaticArray<i32>(0);
  children: Array<Tensor> = new Array<Tensor>(0);
  op: i32 = OP_NONE;
  scalarArg: f32 = 0.0;
  intArg: i32 = 0;
  intArg2: i32 = 0;
  cacheData: StaticArray<f32> = new StaticArray<f32>(0);
}

// ===== Leaf constructors =====

export function tensorFrom(data: StaticArray<f32>, shape: StaticArray<i32>): Tensor {
  const t = new Tensor();
  t.data = data;
  t.grad = new StaticArray<f32>(data.length);
  t.shape = shape;
  return t;
}

export function tensorZeros(shape: StaticArray<i32>): Tensor {
  let size: i32 = 1;
  for (let i: i32 = 0; i < shape.length; i++) {
    size *= shape[i];
  }
  const t = new Tensor();
  t.data = new StaticArray<f32>(size);
  t.grad = new StaticArray<f32>(size);
  t.shape = shape;
  return t;
}

export function tensorScalar(value: f32): Tensor {
  const t = new Tensor();
  const d = new StaticArray<f32>(1);
  d[0] = value;
  t.data = d;
  t.grad = new StaticArray<f32>(1);
  t.shape = new StaticArray<i32>(0);
  return t;
}

// ===== Elementwise forward ops =====

export function add(a: Tensor, b: Tensor): Tensor {
  const n = a.data.length;
  const out = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    out[i] = a.data[i] + b.data[i];
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(a.shape);
  t.children = [a, b];
  t.op = OP_ADD;
  return t;
}

export function neg(a: Tensor): Tensor {
  const n = a.data.length;
  const out = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    out[i] = -a.data[i];
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(a.shape);
  t.children = [a];
  t.op = OP_NEG;
  return t;
}

export function relu(a: Tensor): Tensor {
  const n = a.data.length;
  const out = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    out[i] = a.data[i] > f32(0.0) ? a.data[i] : f32(0.0);
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(a.shape);
  t.children = [a];
  t.op = OP_RELU;
  return t;
}

export function logOp(a: Tensor): Tensor {
  const n = a.data.length;
  const out = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    out[i] = Mathf.log(a.data[i]);
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(a.shape);
  t.children = [a];
  t.op = OP_LOG;
  return t;
}

export function mulScalar(a: Tensor, s: f32): Tensor {
  const n = a.data.length;
  const out = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    out[i] = a.data[i] * s;
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(a.shape);
  t.children = [a];
  t.op = OP_MUL_SCALAR;
  t.scalarArg = s;
  return t;
}

export function divScalar(a: Tensor, s: f32): Tensor {
  const n = a.data.length;
  const out = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    out[i] = a.data[i] / s;
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(a.shape);
  t.children = [a];
  t.op = OP_DIV_SCALAR;
  t.scalarArg = s;
  return t;
}

export function mul(a: Tensor, b: Tensor): Tensor {
  const n = a.data.length;
  const out = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    out[i] = a.data[i] * b.data[i];
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(a.shape);
  t.children = [a, b];
  t.op = OP_MUL;
  return t;
}

// ===== Reduction ops =====

export function tensorSum(a: Tensor): Tensor {
  let s: f32 = 0.0;
  for (let i: i32 = 0; i < a.data.length; i++) {
    s += a.data[i];
  }
  const d = new StaticArray<f32>(1);
  d[0] = s;
  const t = new Tensor();
  t.data = d;
  t.grad = new StaticArray<f32>(1);
  t.shape = new StaticArray<i32>(0);
  t.children = [a];
  t.op = OP_SUM;
  return t;
}

// ===== Structural ops =====

export function slice(a: Tensor, start: i32, end: i32): Tensor {
  const len = end - start;
  const out = new StaticArray<f32>(len);
  for (let i: i32 = 0; i < len; i++) {
    out[i] = a.data[start + i];
  }
  const s = new StaticArray<i32>(1);
  s[0] = len;
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(len);
  t.shape = s;
  t.children = [a];
  t.op = OP_SLICE;
  t.intArg = start;
  t.intArg2 = end;
  return t;
}

export function concat(parts: Array<Tensor>): Tensor {
  let totalLen: i32 = 0;
  for (let i: i32 = 0; i < parts.length; i++) {
    totalLen += parts[i].data.length;
  }
  const out = new StaticArray<f32>(totalLen);
  let offset: i32 = 0;
  for (let i: i32 = 0; i < parts.length; i++) {
    const p = parts[i];
    for (let j: i32 = 0; j < p.data.length; j++) {
      out[offset + j] = p.data[j];
    }
    offset += p.data.length;
  }
  const s = new StaticArray<i32>(1);
  s[0] = totalLen;
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(totalLen);
  t.shape = s;
  t.children = parts;
  t.op = OP_CONCAT;
  return t;
}

export function embedding(table: Tensor, id: i32): Tensor {
  const cols = table.shape[1];
  const out = new StaticArray<f32>(cols);
  const base = id * cols;
  for (let i: i32 = 0; i < cols; i++) {
    out[i] = table.data[base + i];
  }
  const s = new StaticArray<i32>(1);
  s[0] = cols;
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(cols);
  t.shape = s;
  t.children = [table];
  t.op = OP_EMBEDDING;
  t.intArg = id;
  return t;
}

// ===== Composite ops =====

export function softmax(a: Tensor): Tensor {
  const n = a.data.length;
  // Find max for numerical stability
  let maxVal: f32 = a.data[0];
  for (let i: i32 = 1; i < n; i++) {
    if (a.data[i] > maxVal) maxVal = a.data[i];
  }
  // Compute exp(x - max) and sum
  const out = new StaticArray<f32>(n);
  let sumVal: f32 = 0.0;
  for (let i: i32 = 0; i < n; i++) {
    out[i] = Mathf.exp(a.data[i] - maxVal);
    sumVal += out[i];
  }
  // Normalize
  for (let i: i32 = 0; i < n; i++) {
    out[i] = out[i] / sumVal;
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(a.shape);
  t.children = [a];
  t.op = OP_SOFTMAX;
  return t;
}

export function rmsnorm(a: Tensor, eps: f32): Tensor {
  const n = a.data.length;
  // Compute mean of squares
  let ss: f32 = 0.0;
  for (let i: i32 = 0; i < n; i++) {
    ss += a.data[i] * a.data[i];
  }
  ss = ss / f32(n) + eps;
  const rmsScale: f32 = f32(1.0) / Mathf.sqrt(ss);
  // Normalize
  const out = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    out[i] = a.data[i] * rmsScale;
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(a.shape);
  t.children = [a];
  t.op = OP_RMSNORM;
  t.scalarArg = eps;
  return t;
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  // a is [m, k], b is [k], result is [m]
  const m = a.shape[0];
  const k = a.shape[1];
  const out = new StaticArray<f32>(m);
  for (let i: i32 = 0; i < m; i++) {
    let s: f32 = 0.0;
    for (let j: i32 = 0; j < k; j++) {
      s += a.data[i * k + j] * b.data[j];
    }
    out[i] = s;
  }
  const shape = new StaticArray<i32>(1);
  shape[0] = m;
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(m);
  t.shape = shape;
  t.children = [a, b];
  t.op = OP_MATMUL;
  return t;
}

export function crossEntropy(logits: Tensor, targetId: i32): Tensor {
  const n = logits.data.length;
  // Find max for numerical stability
  let maxVal: f32 = logits.data[0];
  for (let i: i32 = 1; i < n; i++) {
    if (logits.data[i] > maxVal) maxVal = logits.data[i];
  }
  // Compute log-sum-exp
  let sumExp: f32 = 0.0;
  for (let i: i32 = 0; i < n; i++) {
    sumExp += Mathf.exp(logits.data[i] - maxVal);
  }
  const logSumExp = Mathf.log(sumExp);
  // Loss = -(logits[target] - max - logSumExp)
  const loss = -(logits.data[targetId] - maxVal - logSumExp);
  const d = new StaticArray<f32>(1);
  d[0] = loss;
  // Store softmax probabilities in cacheData for backward
  const probs = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    probs[i] = Mathf.exp(logits.data[i] - maxVal) / sumExp;
  }
  const t = new Tensor();
  t.data = d;
  t.grad = new StaticArray<f32>(1);
  t.shape = new StaticArray<i32>(0);
  t.children = [logits];
  t.op = OP_CROSS_ENTROPY;
  t.intArg = targetId;
  t.cacheData = probs;
  return t;
}

export function scale(vec: Tensor, scalar: Tensor): Tensor {
  const n = vec.data.length;
  const s = scalar.data[0];
  const out = new StaticArray<f32>(n);
  for (let i: i32 = 0; i < n; i++) {
    out[i] = vec.data[i] * s;
  }
  const t = new Tensor();
  t.data = out;
  t.grad = new StaticArray<f32>(n);
  t.shape = copyShape(vec.shape);
  t.children = [vec, scalar];
  t.op = OP_SCALE;
  return t;
}

// ===== Backward pass =====

export function backward(loss: Tensor): void {
  // 1. Iterative post-order DFS topological sort
  const order = new Array<Tensor>();
  const visited = new Set<usize>();
  const stack = new Array<Tensor>();
  const stackState = new Array<i32>(); // 0 = unexplored, 1 = children pushed
  stack.push(loss);
  stackState.push(0);
  while (stack.length > 0) {
    const idx = stack.length - 1;
    const node = stack[idx];
    const nodeId = changetype<usize>(node);
    if (stackState[idx] == 0) {
      if (visited.has(nodeId)) {
        stack.pop();
        stackState.pop();
        continue;
      }
      visited.add(nodeId);
      stackState[idx] = 1;
      const ch = node.children;
      for (let i: i32 = 0; i < ch.length; i++) {
        const childId = changetype<usize>(ch[i]);
        if (!visited.has(childId)) {
          stack.push(ch[i]);
          stackState.push(0);
        }
      }
    } else {
      stack.pop();
      stackState.pop();
      order.push(node);
    }
  }

  // 2. Set loss gradient to 1.0
  loss.grad[0] = f32(1.0);

  // 3. Reverse walk: dispatch backward for each op
  for (let i: i32 = order.length - 1; i >= 0; i--) {
    backwardOp(order[i]);
  }

  // 4. Detach graph for GC
  for (let i: i32 = 0; i < order.length; i++) {
    order[i].children = new Array<Tensor>(0);
  }
}

function backwardOp(t: Tensor): void {
  const g = t.grad;
  const ch = t.children;
  if (ch.length == 0) return; // leaf

  if (t.op == OP_ADD) {
    const a = ch[0];
    const b = ch[1];
    for (let i: i32 = 0; i < g.length; i++) {
      a.grad[i] += g[i];
      b.grad[i] += g[i];
    }
  } else if (t.op == OP_NEG) {
    const a = ch[0];
    for (let i: i32 = 0; i < g.length; i++) {
      a.grad[i] += -g[i];
    }
  } else if (t.op == OP_RELU) {
    const a = ch[0];
    for (let i: i32 = 0; i < g.length; i++) {
      a.grad[i] += a.data[i] > f32(0.0) ? g[i] : f32(0.0);
    }
  } else if (t.op == OP_LOG) {
    const a = ch[0];
    for (let i: i32 = 0; i < g.length; i++) {
      a.grad[i] += g[i] / a.data[i];
    }
  } else if (t.op == OP_MUL_SCALAR) {
    const a = ch[0];
    const s = t.scalarArg;
    for (let i: i32 = 0; i < a.data.length; i++) {
      a.grad[i] += g[i] * s;
    }
  } else if (t.op == OP_DIV_SCALAR) {
    const a = ch[0];
    const s = t.scalarArg;
    for (let i: i32 = 0; i < a.data.length; i++) {
      a.grad[i] += g[i] / s;
    }
  } else if (t.op == OP_MUL) {
    const a = ch[0];
    const b = ch[1];
    for (let i: i32 = 0; i < g.length; i++) {
      a.grad[i] += g[i] * b.data[i];
      b.grad[i] += g[i] * a.data[i];
    }
  } else if (t.op == OP_SUM) {
    const a = ch[0];
    const gVal = g[0];
    for (let i: i32 = 0; i < a.data.length; i++) {
      a.grad[i] += gVal;
    }
  } else if (t.op == OP_SLICE) {
    const a = ch[0];
    const start = t.intArg;
    const len = t.data.length;
    for (let i: i32 = 0; i < len; i++) {
      a.grad[start + i] += g[i];
    }
  } else if (t.op == OP_CONCAT) {
    let offset: i32 = 0;
    for (let ci: i32 = 0; ci < ch.length; ci++) {
      const child = ch[ci];
      const childLen = child.data.length;
      for (let j: i32 = 0; j < childLen; j++) {
        child.grad[j] += g[offset + j];
      }
      offset += childLen;
    }
  } else if (t.op == OP_EMBEDDING) {
    const table = ch[0];
    const id = t.intArg;
    const cols = table.shape[1];
    const base = id * cols;
    for (let i: i32 = 0; i < t.data.length; i++) {
      table.grad[base + i] += g[i];
    }
  } else if (t.op == OP_MATMUL) {
    // a is [m, k], b is [k], result is [m]
    const a = ch[0];
    const b = ch[1];
    const m = a.shape[0];
    const k = a.shape[1];
    for (let i: i32 = 0; i < m; i++) {
      for (let j: i32 = 0; j < k; j++) {
        a.grad[i * k + j] += g[i] * b.data[j];
        b.grad[j] += a.data[i * k + j] * g[i];
      }
    }
  } else if (t.op == OP_SOFTMAX) {
    // Jacobian-vector product: out.grad[i] = out.data[i] * (g[i] - dot(g, out.data))
    const a = ch[0];
    const n = t.data.length;
    let dotVal: f32 = 0.0;
    for (let i: i32 = 0; i < n; i++) {
      dotVal += g[i] * t.data[i];
    }
    for (let i: i32 = 0; i < n; i++) {
      a.grad[i] += t.data[i] * (g[i] - dotVal);
    }
  } else if (t.op == OP_RMSNORM) {
    // Forward: out[i] = x[i] / rms, where rms = sqrt(mean(x^2) + eps)
    // Backward: dx[i] = (1/rms) * (g[i] - out[i] * dot(g, out) / n)
    const a = ch[0];
    const n = a.data.length;
    const eps = t.scalarArg;
    // Recompute rms from input
    let ss: f32 = 0.0;
    for (let i: i32 = 0; i < n; i++) {
      ss += a.data[i] * a.data[i];
    }
    const rmsScale: f32 = f32(1.0) / Mathf.sqrt(ss / f32(n) + eps);
    // dot(g, out)
    let dotGOut: f32 = 0.0;
    for (let i: i32 = 0; i < n; i++) {
      dotGOut += g[i] * t.data[i];
    }
    for (let i: i32 = 0; i < n; i++) {
      a.grad[i] += rmsScale * (g[i] - t.data[i] * dotGOut / f32(n));
    }
  } else if (t.op == OP_CROSS_ENTROPY) {
    // Backward: logits.grad[i] += upstream * (probs[i] - (i == target ? 1 : 0))
    const logits = ch[0];
    const targetId = t.intArg;
    const probs = t.cacheData;
    const upstream = g[0];
    for (let i: i32 = 0; i < logits.data.length; i++) {
      const indicator: f32 = i == targetId ? f32(1.0) : f32(0.0);
      logits.grad[i] += upstream * (probs[i] - indicator);
    }
  } else if (t.op == OP_SCALE) {
    // vec * scalar tensor
    const vec = ch[0];
    const scalar = ch[1];
    const s = scalar.data[0];
    let dotVal: f32 = 0.0;
    for (let i: i32 = 0; i < vec.data.length; i++) {
      vec.grad[i] += g[i] * s;
      dotVal += g[i] * vec.data[i];
    }
    scalar.grad[0] += dotVal;
  }
}

// ===== Helper =====

function copyShape(shape: StaticArray<i32>): StaticArray<i32> {
  const s = new StaticArray<i32>(shape.length);
  for (let i: i32 = 0; i < shape.length; i++) {
    s[i] = shape[i];
  }
  return s;
}
