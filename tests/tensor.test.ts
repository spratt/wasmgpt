import { test, expect } from "assemblyscript-unittest-framework/assembly";
import {
  Tensor, tensorFrom, tensorZeros, tensorScalar,
  add, neg, relu, logOp, mulScalar, divScalar, mul,
  tensorSum, slice, concat, embedding, softmax, rmsnorm,
  matmul, crossEntropy, scale, backward,
} from "../src/tensor";

// ===== Helper: compare f32 with tolerance =====

function approx(actual: f32, expected: f32, tol: f32 = f32(1e-4)): bool {
  const diff = actual - expected;
  return (diff < tol) && (diff > -tol);
}

// ===== Helper: finite-difference gradient check =====
// Returns the numerical gradient of a scalar-output function w.r.t. input[idx]
// fn takes an input tensor and returns a scalar tensor

// ===== Leaf constructors =====

test("tensorFrom creates correct tensor", () => {
  const data = StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0)]);
  const shape = StaticArray.fromArray<i32>([3]);
  const t = tensorFrom(data, shape);
  expect(t.data.length).equal(3);
  expect(t.grad.length).equal(3);
  expect(t.shape[0]).equal(3);
  expect(t.data[0]).equal(f32(1.0));
  expect(t.data[2]).equal(f32(3.0));
  expect(t.grad[0]).equal(f32(0.0));
  expect(t.children.length).equal(0);
});

test("tensorZeros creates zero tensor", () => {
  const shape = StaticArray.fromArray<i32>([2, 3]);
  const t = tensorZeros(shape);
  expect(t.data.length).equal(6);
  expect(t.grad.length).equal(6);
  expect(t.shape.length).equal(2);
  expect(t.shape[0]).equal(2);
  expect(t.shape[1]).equal(3);
  for (let i: i32 = 0; i < 6; i++) {
    expect(t.data[i]).equal(f32(0.0));
  }
});

test("tensorScalar creates scalar", () => {
  const t = tensorScalar(f32(3.14));
  expect(t.data.length).equal(1);
  expect(t.data[0]).equal(f32(3.14));
  expect(t.shape.length).equal(0);
});

// ===== Forward: add =====

test("add forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0)]),
                        StaticArray.fromArray<i32>([2]));
  const b = tensorFrom(StaticArray.fromArray<f32>([f32(3.0), f32(4.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = add(a, b);
  expect(c.data[0]).equal(f32(4.0));
  expect(c.data[1]).equal(f32(6.0));
});

// ===== Forward: neg =====

test("neg forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(-2.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = neg(a);
  expect(c.data[0]).equal(f32(-1.0));
  expect(c.data[1]).equal(f32(2.0));
});

// ===== Forward: relu =====

test("relu forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(3.0), f32(-1.0), f32(0.0), f32(2.0)]),
                        StaticArray.fromArray<i32>([4]));
  const c = relu(a);
  expect(c.data[0]).equal(f32(3.0));
  expect(c.data[1]).equal(f32(0.0));
  expect(c.data[2]).equal(f32(0.0));
  expect(c.data[3]).equal(f32(2.0));
});

// ===== Forward: logOp =====

test("logOp forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.718281828)]),
                        StaticArray.fromArray<i32>([2]));
  const c = logOp(a);
  expect(approx(c.data[0], f32(0.0))).equal(true);
  expect(approx(c.data[1], f32(1.0), f32(1e-3))).equal(true);
});

// ===== Forward: mulScalar =====

test("mulScalar forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(2.0), f32(3.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = mulScalar(a, f32(0.5));
  expect(c.data[0]).equal(f32(1.0));
  expect(c.data[1]).equal(f32(1.5));
});

// ===== Forward: divScalar =====

test("divScalar forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(6.0), f32(9.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = divScalar(a, f32(3.0));
  expect(c.data[0]).equal(f32(2.0));
  expect(c.data[1]).equal(f32(3.0));
});

// ===== Forward: mul (elementwise) =====

test("mul forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(2.0), f32(3.0)]),
                        StaticArray.fromArray<i32>([2]));
  const b = tensorFrom(StaticArray.fromArray<f32>([f32(4.0), f32(5.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = mul(a, b);
  expect(c.data[0]).equal(f32(8.0));
  expect(c.data[1]).equal(f32(15.0));
});

// ===== Forward: tensorSum =====

test("tensorSum forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0)]),
                        StaticArray.fromArray<i32>([3]));
  const c = tensorSum(a);
  expect(c.data[0]).equal(f32(6.0));
  expect(c.shape.length).equal(0);
});

// ===== Forward: slice =====

test("slice forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(10.0), f32(20.0), f32(30.0), f32(40.0)]),
                        StaticArray.fromArray<i32>([4]));
  const c = slice(a, 1, 3);
  expect(c.data.length).equal(2);
  expect(c.data[0]).equal(f32(20.0));
  expect(c.data[1]).equal(f32(30.0));
});

// ===== Forward: concat =====

test("concat forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0)]),
                        StaticArray.fromArray<i32>([2]));
  const b = tensorFrom(StaticArray.fromArray<f32>([f32(3.0)]),
                        StaticArray.fromArray<i32>([1]));
  const c = concat([a, b]);
  expect(c.data.length).equal(3);
  expect(c.data[0]).equal(f32(1.0));
  expect(c.data[1]).equal(f32(2.0));
  expect(c.data[2]).equal(f32(3.0));
});

// ===== Forward: embedding =====

test("embedding forward", () => {
  // 3x2 table: row 0 = [1,2], row 1 = [3,4], row 2 = [5,6]
  const table = tensorFrom(
    StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0), f32(4.0), f32(5.0), f32(6.0)]),
    StaticArray.fromArray<i32>([3, 2])
  );
  const c = embedding(table, 1);
  expect(c.data.length).equal(2);
  expect(c.data[0]).equal(f32(3.0));
  expect(c.data[1]).equal(f32(4.0));
});

// ===== Forward: softmax =====

test("softmax sums to 1", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0)]),
                        StaticArray.fromArray<i32>([3]));
  const c = softmax(a);
  let total: f32 = 0.0;
  for (let i: i32 = 0; i < 3; i++) {
    total += c.data[i];
  }
  expect(approx(total, f32(1.0))).equal(true);
});

test("softmax max element has largest prob", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(5.0), f32(2.0)]),
                        StaticArray.fromArray<i32>([3]));
  const c = softmax(a);
  expect(c.data[1] > c.data[0]).equal(true);
  expect(c.data[1] > c.data[2]).equal(true);
});

test("softmax stability with large values", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(100.0), f32(101.0), f32(100.0)]),
                        StaticArray.fromArray<i32>([3]));
  const c = softmax(a);
  let total: f32 = 0.0;
  for (let i: i32 = 0; i < 3; i++) {
    total += c.data[i];
    expect(isFinite(c.data[i])).equal(true);
  }
  expect(approx(total, f32(1.0))).equal(true);
});

// ===== Forward: rmsnorm =====

test("rmsnorm forward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(3.0), f32(4.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = rmsnorm(a, f32(1e-5));
  // rms = sqrt((9+16)/2 + 1e-5) = sqrt(12.5 + 1e-5) ≈ 3.5355
  // out[0] = 3/3.5355 ≈ 0.8485, out[1] = 4/3.5355 ≈ 1.1314
  expect(approx(c.data[0], f32(0.8485), f32(1e-3))).equal(true);
  expect(approx(c.data[1], f32(1.1314), f32(1e-3))).equal(true);
});

// ===== Forward: matmul =====

test("matmul 2x3 times 3-vector", () => {
  // [[1,2,3],[4,5,6]] * [1,0,1] = [4, 10]
  const a = tensorFrom(
    StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0), f32(4.0), f32(5.0), f32(6.0)]),
    StaticArray.fromArray<i32>([2, 3])
  );
  const b = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(0.0), f32(1.0)]),
                        StaticArray.fromArray<i32>([3]));
  const c = matmul(a, b);
  expect(c.data.length).equal(2);
  expect(c.data[0]).equal(f32(4.0));
  expect(c.data[1]).equal(f32(10.0));
});

// ===== Forward: crossEntropy =====

test("crossEntropy forward", () => {
  // logits = [1, 2, 3], target = 2
  // softmax(1,2,3) = [0.0900, 0.2447, 0.6652]
  // loss = -log(0.6652) ≈ 0.4076
  const logits = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0)]),
                              StaticArray.fromArray<i32>([3]));
  const c = crossEntropy(logits, 2);
  expect(c.data.length).equal(1);
  expect(approx(c.data[0], f32(0.4076), f32(1e-3))).equal(true);
  // Check that cacheData has probs summing to 1
  let total: f32 = 0.0;
  for (let i: i32 = 0; i < 3; i++) {
    total += c.cacheData[i];
  }
  expect(approx(total, f32(1.0))).equal(true);
});

// ===== Forward: scale =====

test("scale forward", () => {
  const vec = tensorFrom(StaticArray.fromArray<f32>([f32(2.0), f32(3.0), f32(4.0)]),
                          StaticArray.fromArray<i32>([3]));
  const s = tensorScalar(f32(0.5));
  const c = scale(vec, s);
  expect(c.data[0]).equal(f32(1.0));
  expect(c.data[1]).equal(f32(1.5));
  expect(c.data[2]).equal(f32(2.0));
});

// ===== Backward: add =====

test("add backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0)]),
                        StaticArray.fromArray<i32>([2]));
  const b = tensorFrom(StaticArray.fromArray<f32>([f32(3.0), f32(4.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = tensorSum(add(a, b));
  backward(c);
  // d(sum(a+b))/da = 1, d(sum(a+b))/db = 1
  expect(a.grad[0]).equal(f32(1.0));
  expect(a.grad[1]).equal(f32(1.0));
  expect(b.grad[0]).equal(f32(1.0));
  expect(b.grad[1]).equal(f32(1.0));
});

// ===== Backward: neg =====

test("neg backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = tensorSum(neg(a));
  backward(c);
  expect(a.grad[0]).equal(f32(-1.0));
  expect(a.grad[1]).equal(f32(-1.0));
});

// ===== Backward: relu =====

test("relu backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(3.0), f32(-1.0), f32(0.0), f32(2.0)]),
                        StaticArray.fromArray<i32>([4]));
  const c = tensorSum(relu(a));
  backward(c);
  expect(a.grad[0]).equal(f32(1.0));  // positive -> gradient passes through
  expect(a.grad[1]).equal(f32(0.0));  // negative -> gradient blocked
  expect(a.grad[2]).equal(f32(0.0));  // zero -> gradient blocked
  expect(a.grad[3]).equal(f32(1.0));
});

// ===== Backward: logOp =====

test("logOp backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(2.0), f32(4.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = tensorSum(logOp(a));
  backward(c);
  // d(log(x))/dx = 1/x
  expect(approx(a.grad[0], f32(0.5))).equal(true);
  expect(approx(a.grad[1], f32(0.25))).equal(true);
});

// ===== Backward: mulScalar =====

test("mulScalar backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(2.0), f32(3.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = tensorSum(mulScalar(a, f32(5.0)));
  backward(c);
  expect(a.grad[0]).equal(f32(5.0));
  expect(a.grad[1]).equal(f32(5.0));
});

// ===== Backward: divScalar =====

test("divScalar backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(6.0), f32(9.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = tensorSum(divScalar(a, f32(3.0)));
  backward(c);
  expect(approx(a.grad[0], f32(1.0 / 3.0))).equal(true);
  expect(approx(a.grad[1], f32(1.0 / 3.0))).equal(true);
});

// ===== Backward: mul (elementwise) =====

test("mul backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(2.0), f32(3.0)]),
                        StaticArray.fromArray<i32>([2]));
  const b = tensorFrom(StaticArray.fromArray<f32>([f32(4.0), f32(5.0)]),
                        StaticArray.fromArray<i32>([2]));
  const c = tensorSum(mul(a, b));
  backward(c);
  // d(sum(a*b))/da[i] = b[i], d/db[i] = a[i]
  expect(a.grad[0]).equal(f32(4.0));
  expect(a.grad[1]).equal(f32(5.0));
  expect(b.grad[0]).equal(f32(2.0));
  expect(b.grad[1]).equal(f32(3.0));
});

// ===== Backward: tensorSum =====

test("tensorSum backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0)]),
                        StaticArray.fromArray<i32>([3]));
  const c = tensorSum(a);
  backward(c);
  for (let i: i32 = 0; i < 3; i++) {
    expect(a.grad[i]).equal(f32(1.0));
  }
});

// ===== Backward: slice =====

test("slice backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(10.0), f32(20.0), f32(30.0), f32(40.0)]),
                        StaticArray.fromArray<i32>([4]));
  const c = tensorSum(slice(a, 1, 3));
  backward(c);
  expect(a.grad[0]).equal(f32(0.0));
  expect(a.grad[1]).equal(f32(1.0));
  expect(a.grad[2]).equal(f32(1.0));
  expect(a.grad[3]).equal(f32(0.0));
});

// ===== Backward: concat =====

test("concat backward", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0)]),
                        StaticArray.fromArray<i32>([2]));
  const b = tensorFrom(StaticArray.fromArray<f32>([f32(3.0)]),
                        StaticArray.fromArray<i32>([1]));
  const c = tensorSum(concat([a, b]));
  backward(c);
  expect(a.grad[0]).equal(f32(1.0));
  expect(a.grad[1]).equal(f32(1.0));
  expect(b.grad[0]).equal(f32(1.0));
});

// ===== Backward: embedding =====

test("embedding backward", () => {
  const table = tensorFrom(
    StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0), f32(4.0), f32(5.0), f32(6.0)]),
    StaticArray.fromArray<i32>([3, 2])
  );
  const c = tensorSum(embedding(table, 1));
  backward(c);
  // Only row 1 should have gradient
  expect(table.grad[0]).equal(f32(0.0));
  expect(table.grad[1]).equal(f32(0.0));
  expect(table.grad[2]).equal(f32(1.0));
  expect(table.grad[3]).equal(f32(1.0));
  expect(table.grad[4]).equal(f32(0.0));
  expect(table.grad[5]).equal(f32(0.0));
});

// ===== Backward: matmul (finite difference) =====

test("matmul backward", () => {
  // W = [[1,2],[3,4]], x = [5,6], Wx = [17, 39]
  const w = tensorFrom(
    StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0), f32(4.0)]),
    StaticArray.fromArray<i32>([2, 2])
  );
  const x = tensorFrom(StaticArray.fromArray<f32>([f32(5.0), f32(6.0)]),
                        StaticArray.fromArray<i32>([2]));
  const loss = tensorSum(matmul(w, x));
  backward(loss);
  // d(sum(Wx))/dW[i][j] = x[j] (summed over output dim)
  // d(sum(Wx))/dW[0][0] = x[0] = 5, dW[0][1] = x[1] = 6
  // d(sum(Wx))/dW[1][0] = x[0] = 5, dW[1][1] = x[1] = 6
  expect(w.grad[0]).equal(f32(5.0));
  expect(w.grad[1]).equal(f32(6.0));
  expect(w.grad[2]).equal(f32(5.0));
  expect(w.grad[3]).equal(f32(6.0));
  // d(sum(Wx))/dx[j] = sum_i(W[i][j]) = W[0][j] + W[1][j]
  // dx[0] = 1+3 = 4, dx[1] = 2+4 = 6
  expect(x.grad[0]).equal(f32(4.0));
  expect(x.grad[1]).equal(f32(6.0));
});

// ===== Backward: softmax (finite difference) =====

test("softmax backward finite diff", () => {
  const eps: f32 = f32(1e-3);
  const tol: f32 = f32(5e-2);
  const vals = StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0)]);
  // Compute analytic gradient: d(sum(softmax(x)))/dx = 0 (softmax sums to 1)
  // But that's trivial. Instead test: d(softmax(x)[1])/dx
  // Use: loss = slice(softmax(x), 1, 2) -> sum -> scalar
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0)]),
                        StaticArray.fromArray<i32>([3]));
  const loss = tensorSum(slice(softmax(a), 1, 2));
  backward(loss);
  // Finite difference for each input
  for (let i: i32 = 0; i < 3; i++) {
    const saved = vals[i];
    // f(x + eps)
    const aPlus = tensorFrom(StaticArray.fromArray<f32>([vals[0], vals[1], vals[2]]),
                              StaticArray.fromArray<i32>([3]));
    aPlus.data[i] = saved + eps;
    const lPlus = tensorSum(slice(softmax(aPlus), 1, 2));
    // f(x - eps)
    const aMinus = tensorFrom(StaticArray.fromArray<f32>([vals[0], vals[1], vals[2]]),
                                StaticArray.fromArray<i32>([3]));
    aMinus.data[i] = saved - eps;
    const lMinus = tensorSum(slice(softmax(aMinus), 1, 2));
    const numGrad = (lPlus.data[0] - lMinus.data[0]) / (f32(2.0) * eps);
    expect(approx(a.grad[i], numGrad, tol)).equal(true);
  }
});

// ===== Backward: rmsnorm (finite difference) =====

test("rmsnorm backward finite diff", () => {
  const eps: f32 = f32(1e-3);
  const tol: f32 = f32(5e-2);
  const vals = StaticArray.fromArray<f32>([f32(3.0), f32(4.0)]);
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(3.0), f32(4.0)]),
                        StaticArray.fromArray<i32>([2]));
  const loss = tensorSum(rmsnorm(a, f32(1e-5)));
  backward(loss);
  for (let i: i32 = 0; i < 2; i++) {
    const saved = vals[i];
    const aPlus = tensorFrom(StaticArray.fromArray<f32>([vals[0], vals[1]]),
                              StaticArray.fromArray<i32>([2]));
    aPlus.data[i] = saved + eps;
    const lPlus = tensorSum(rmsnorm(aPlus, f32(1e-5)));
    const aMinus = tensorFrom(StaticArray.fromArray<f32>([vals[0], vals[1]]),
                                StaticArray.fromArray<i32>([2]));
    aMinus.data[i] = saved - eps;
    const lMinus = tensorSum(rmsnorm(aMinus, f32(1e-5)));
    const numGrad = (lPlus.data[0] - lMinus.data[0]) / (f32(2.0) * eps);
    expect(approx(a.grad[i], numGrad, tol)).equal(true);
  }
});

// ===== Backward: crossEntropy =====

test("crossEntropy backward finite diff", () => {
  const eps: f32 = f32(1e-3);
  const tol: f32 = f32(5e-2);
  const vals = StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0)]);
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(2.0), f32(3.0)]),
                        StaticArray.fromArray<i32>([3]));
  const loss = crossEntropy(a, 1);
  backward(loss);
  for (let i: i32 = 0; i < 3; i++) {
    const saved = vals[i];
    const aPlus = tensorFrom(StaticArray.fromArray<f32>([vals[0], vals[1], vals[2]]),
                              StaticArray.fromArray<i32>([3]));
    aPlus.data[i] = saved + eps;
    const lPlus = crossEntropy(aPlus, 1);
    const aMinus = tensorFrom(StaticArray.fromArray<f32>([vals[0], vals[1], vals[2]]),
                                StaticArray.fromArray<i32>([3]));
    aMinus.data[i] = saved - eps;
    const lMinus = crossEntropy(aMinus, 1);
    const numGrad = (lPlus.data[0] - lMinus.data[0]) / (f32(2.0) * eps);
    expect(approx(a.grad[i], numGrad, tol)).equal(true);
  }
});

// ===== Backward: scale =====

test("scale backward", () => {
  const vec = tensorFrom(StaticArray.fromArray<f32>([f32(2.0), f32(3.0)]),
                          StaticArray.fromArray<i32>([2]));
  const s = tensorScalar(f32(4.0));
  const loss = tensorSum(scale(vec, s));
  backward(loss);
  // d(sum(vec * s))/dvec[i] = s = 4
  expect(vec.grad[0]).equal(f32(4.0));
  expect(vec.grad[1]).equal(f32(4.0));
  // d(sum(vec * s))/ds = sum(vec) = 5
  expect(s.grad[0]).equal(f32(5.0));
});

// ===== Composed: matmul + add + relu + matmul =====

test("composed matmul+add+relu+matmul backward", () => {
  // W1 = [[1,0],[0,1]], b = [0.5, -0.5], W2 = [[1,1]]
  // x = [1, -1]
  // h = W1*x + b = [1.5, -1.5]
  // r = relu(h) = [1.5, 0]
  // y = W2*r = [1.5]
  // loss = sum(y) = 1.5
  const w1 = tensorFrom(
    StaticArray.fromArray<f32>([f32(1.0), f32(0.0), f32(0.0), f32(1.0)]),
    StaticArray.fromArray<i32>([2, 2])
  );
  const b = tensorFrom(StaticArray.fromArray<f32>([f32(0.5), f32(-0.5)]),
                        StaticArray.fromArray<i32>([2]));
  const w2 = tensorFrom(
    StaticArray.fromArray<f32>([f32(1.0), f32(1.0)]),
    StaticArray.fromArray<i32>([1, 2])
  );
  const x = tensorFrom(StaticArray.fromArray<f32>([f32(1.0), f32(-1.0)]),
                        StaticArray.fromArray<i32>([2]));
  const h = add(matmul(w1, x), b);
  const r = relu(h);
  const y = matmul(w2, r);
  const loss = tensorSum(y);
  backward(loss);
  // Verify loss value
  expect(approx(loss.data[0], f32(1.5))).equal(true);
  // x should have non-zero gradients
  expect(x.grad[0] != f32(0.0)).equal(true);
  // w1, w2, b should have non-zero gradients
  expect(w1.grad[0] != f32(0.0)).equal(true);
  expect(w2.grad[0] != f32(0.0)).equal(true);
  expect(b.grad[0] != f32(0.0)).equal(true);
});

// ===== Graph detachment =====

test("backward detaches graph", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(1.0)]),
                        StaticArray.fromArray<i32>([1]));
  const b = tensorFrom(StaticArray.fromArray<f32>([f32(2.0)]),
                        StaticArray.fromArray<i32>([1]));
  const c = add(a, b);
  const loss = tensorSum(c);
  expect(c.children.length).equal(2);
  expect(loss.children.length).equal(1);
  backward(loss);
  // After backward, all children should be empty
  expect(c.children.length).equal(0);
  expect(loss.children.length).equal(0);
});

// ===== Gradient accumulation: tensor used twice =====

test("gradient accumulates when tensor used twice", () => {
  const a = tensorFrom(StaticArray.fromArray<f32>([f32(3.0)]),
                        StaticArray.fromArray<i32>([1]));
  // loss = sum(a + a) = 2*a = 6
  const c = add(a, a);
  const loss = tensorSum(c);
  backward(loss);
  // d(2a)/da = 2
  expect(a.grad[0]).equal(f32(2.0));
});
