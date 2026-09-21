import test from 'node:test';
import assert from 'node:assert/strict';
import {
  weightBytes, kvBytesPerToken, vramUsage, prefillMs,
  tokenIntervalMs, validate, simulate, GPU_PRESETS, MODEL_PRESETS,
} from '../sim.js';

const close = (actual, expected, tol = 0.01) =>
  assert.ok(Math.abs(actual - expected) <= tol, `expected ${expected} ± ${tol}, got ${actual}`);

const base = {
  vramGB: 24, gpuCount: 1, bandwidthGBs: 1008, tflops: 165,
  paramsB: 8, bits: 16, promptTokens: 1000, outputTokens: 1000, batch: 1,
};

test('weightBytes: 8B FP16 = 16e9, INT4 = 4e9', () => {
  assert.equal(weightBytes({ paramsB: 8, bits: 16 }), 16e9);
  assert.equal(weightBytes({ paramsB: 8, bits: 4 }), 4e9);
});

test('kvBytesPerToken: 아키텍처가 있으면 정확한 공식, 없으면 근사', () => {
  assert.equal(kvBytesPerToken({ paramsB: 8.03, layers: 32, kvHeads: 8, headDim: 128 }), 131072);
  assert.equal(kvBytesPerToken({ paramsB: 8 }), 131072);
  close(kvBytesPerToken({ paramsB: 32 }), 262144);
});

test('vramUsage: 가중치 + KV + 10% 오버헤드', () => {
  const u = vramUsage(base);
  assert.equal(u.weights, 16e9);
  assert.equal(u.kv, 131072 * 2000);
  close(u.overhead, (u.weights + u.kv) * 0.1, 1);
  close(u.total, (u.weights + u.kv) * 1.1, 1);
});

test('vramUsage: 배치가 KV를 배수로 늘린다', () => {
  assert.equal(vramUsage({ ...base, batch: 4 }).kv, 4 * vramUsage(base).kv);
});

test('prefillMs: compute-bound (H100, 8B, 2000 토큰) = 64.71 + 20 ms', () => {
  const input = { ...base, bandwidthGBs: 3350, tflops: 989, promptTokens: 2000 };
  close(prefillMs(input), 64.71 + 20);
});

test('prefillMs: 프롬프트가 아주 짧으면 가중치 읽기 시간이 하한', () => {
  const input = { ...base, promptTokens: 1 };
  // 16e9 / (1008e9 * 0.8) = 19.841 ms
  close(prefillMs(input), 19.841 + 20);
});

test('tokenIntervalMs: 4090, 8B FP16, ctx 0 = 19.841 ms (≈50 TPS)', () => {
  close(tokenIntervalMs(base, 0), 19.841);
});

test('tokenIntervalMs: 컨텍스트가 길수록 느리다', () => {
  assert.ok(tokenIntervalMs(base, 8000) > tokenIntervalMs(base, 0));
});

test('simulate: 배치를 키우면 총 처리량은 늘고 사용자당 TPS는 줄어든다', () => {
  const one = simulate(base);
  const eight = simulate({ ...base, batch: 8 });
  assert.ok(eight.tpsTotal > one.tpsTotal);
  assert.ok(eight.tpsPerUser < one.tpsPerUser);
});

test('simulate: 양자화가 낮을수록 TPS가 높다', () => {
  assert.ok(simulate({ ...base, bits: 4 }).tpsPerUser > simulate({ ...base, bits: 16 }).tpsPerUser);
});

test('simulate: 컨텍스트가 늘면 끝 시점 TPS가 시작 시점보다 낮다', () => {
  const r = simulate(base);
  assert.ok(r.tpsPerUserEnd < r.tpsPerUser);
});

test('simulate: OOM 판정 (70B FP16 / 24GB → OOM, 8B INT4 → OK)', () => {
  assert.equal(simulate({ ...base, paramsB: 70 }).oom, true);
  assert.equal(simulate({ ...base, bits: 4 }).oom, false);
});

test('simulate: GPU 개수를 늘리면 용량과 속도가 늘어난다', () => {
  const one = simulate(base);
  const two = simulate({ ...base, gpuCount: 2 });
  assert.equal(two.capacityBytes, 2 * one.capacityBytes);
  assert.ok(two.tpsPerUser > one.tpsPerUser);
});

test('validate: 정상 입력은 오류 없음', () => {
  assert.deepEqual(validate(base), {});
});

test('validate: 0 이하/NaN/비정수/잘못된 bits를 잡는다', () => {
  const errors = validate({
    ...base, vramGB: 0, gpuCount: 1.5, bandwidthGBs: NaN, tflops: -1,
    paramsB: 0, bits: 3, promptTokens: 0, outputTokens: 0, batch: 0,
  });
  for (const k of ['vramGB', 'gpuCount', 'bandwidthGBs', 'tflops', 'paramsB', 'bits', 'promptTokens', 'outputTokens', 'batch']) {
    assert.ok(errors[k], `${k} 오류가 있어야 함`);
  }
});

test('프리셋: 마지막 항목은 custom, 나머지는 수치가 유효', () => {
  assert.equal(GPU_PRESETS.at(-1).id, 'custom');
  assert.equal(MODEL_PRESETS.at(-1).id, 'custom');
  for (const g of GPU_PRESETS.slice(0, -1)) {
    assert.ok(g.vramGB > 0 && g.bandwidthGBs > 0 && g.tflops > 0, g.id);
  }
  for (const m of MODEL_PRESETS.slice(0, -1)) {
    assert.ok(m.paramsB > 0 && m.layers > 0 && m.kvHeads > 0 && m.headDim > 0, m.id);
  }
});
