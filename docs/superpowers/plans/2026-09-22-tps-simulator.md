# LLM TPS Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GPU 사양과 모델 설정으로 LLM 추론의 TTFT/TPS를 roofline 모델로 계산하고 토큰 출력 애니메이션으로 보여주는 정적 페이지를 만들어 GitHub Pages로 배포한다.

**Architecture:** 바닐라 HTML/JS/CSS, 빌드 없음. 계산은 DOM 의존이 없는 순수 ES module(`sim.js`)로 분리해 `node --test`로 검증하고, `app.js`가 폼 바인딩/렌더링/애니메이션을 맡는다. `main` 브랜치 루트를 Pages 소스로 쓴다.

**Tech Stack:** HTML5, ES modules, CSS, Node(`node --test`, 테스트 전용), Playwright MCP(UI 확인), GitHub Pages

**Spec:** `docs/superpowers/specs/2026-09-22-tps-simulator-design.md`

## Global Constraints

- 빌드 도구/외부 의존성(npm 패키지, CDN) 없음. 브라우저 코드는 `<script type="module">`
- 파일 구성: `index.html`, `sim.js`, `app.js`, `style.css`, `tests/sim.test.js`, `package.json`
- `sim.js`는 DOM/window에 의존하지 않는다
- 결과 화면에 "이론적 추정치이며 실제 추론 엔진에서는 보통 이보다 낮게 나온다"는 안내 문구를 표시한다
- UI 텍스트는 한국어, 코드 식별자는 영어
- VRAM 용량은 GiB(2^30) 기준이며 화면에는 "GB"로 표기한다
- 커밋 메시지 끝에 `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` 를 붙인다
- 배포 URL: https://bluerose7112.github.io/tps_sim/

---

### Task 1: 계산 모델 (`sim.js`) + 단위 테스트

**Files:**
- Create: `package.json`
- Create: `sim.js`
- Test: `tests/sim.test.js`

**Interfaces:**
- Produces (Task 2가 import 한다):
  - `GPU_PRESETS: Array<{id, name, vramGB, bandwidthGBs, tflops}>` (마지막 항목 `id:'custom'`)
  - `MODEL_PRESETS: Array<{id, name, paramsB, layers, kvHeads, headDim}>` (마지막 항목 `id:'custom'`, 수치 필드는 `null`)
  - `input` 객체: `{vramGB, gpuCount, bandwidthGBs, tflops, paramsB, layers?, kvHeads?, headDim?, bits, promptTokens, outputTokens, batch}` (`bits`는 4|8|16)
  - `weightBytes(input): number`
  - `kvBytesPerToken(input): number`
  - `vramUsage(input): {weights, kv, overhead, total}` (bytes, 컨텍스트 = promptTokens + outputTokens 기준)
  - `prefillMs(input): number`
  - `tokenIntervalMs(input, contextTokens): number`
  - `validate(input): Object<string,string>` (필드 id → 오류 메시지, 없으면 `{}`)
  - `simulate(input): {usage, capacityBytes, oom, ttftMs, tpsPerUser, tpsPerUserEnd, tpsTotal}`

- [ ] **Step 1: `package.json` 작성**

```json
{
  "name": "tps-sim",
  "private": true,
  "type": "module",
  "scripts": {
    "test": "node --test"
  }
}
```

- [ ] **Step 2: 실패하는 테스트 작성** — `tests/sim.test.js`

```js
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
```

- [ ] **Step 3: 테스트가 실패하는지 확인**

Run: `cd /home/smallcloud/work/git/tps_sim && npm test`
Expected: FAIL (`Cannot find module '.../sim.js'`)

- [ ] **Step 4: `sim.js` 구현**

```js
// LLM 추론 속도 계산 모델 (roofline 기반). DOM 의존 없음.

export const MEM_EFFICIENCY = 0.8;
export const COMPUTE_EFFICIENCY = 0.5;
export const PREFILL_OVERHEAD_MS = 20;
export const VRAM_OVERHEAD_RATIO = 0.1;
const GIB = 2 ** 30;

// 대역폭: GB/s, tflops: FP16 dense Tensor TFLOPS (공개 사양 기준)
export const GPU_PRESETS = [
  { id: 'rtx3090', name: 'RTX 3090 (24GB)', vramGB: 24, bandwidthGBs: 936, tflops: 71 },
  { id: 'rtx4090', name: 'RTX 4090 (24GB)', vramGB: 24, bandwidthGBs: 1008, tflops: 165 },
  { id: 'rtx5090', name: 'RTX 5090 (32GB)', vramGB: 32, bandwidthGBs: 1792, tflops: 209 },
  { id: 'l40s', name: 'L40S (48GB)', vramGB: 48, bandwidthGBs: 864, tflops: 362 },
  { id: 'a100', name: 'A100 (80GB)', vramGB: 80, bandwidthGBs: 2039, tflops: 312 },
  { id: 'h100', name: 'H100 SXM (80GB)', vramGB: 80, bandwidthGBs: 3350, tflops: 989 },
  { id: 'custom', name: '직접 입력', vramGB: null, bandwidthGBs: null, tflops: null },
];

export const MODEL_PRESETS = [
  { id: 'llama32-3b', name: 'Llama 3.2 3B', paramsB: 3.21, layers: 28, kvHeads: 8, headDim: 128 },
  { id: 'qwen25-7b', name: 'Qwen2.5 7B', paramsB: 7.6, layers: 28, kvHeads: 4, headDim: 128 },
  { id: 'mistral-7b', name: 'Mistral 7B', paramsB: 7.24, layers: 32, kvHeads: 8, headDim: 128 },
  { id: 'llama31-8b', name: 'Llama 3.1 8B', paramsB: 8.03, layers: 32, kvHeads: 8, headDim: 128 },
  { id: 'qwen25-32b', name: 'Qwen2.5 32B', paramsB: 32.5, layers: 64, kvHeads: 8, headDim: 128 },
  { id: 'llama31-70b', name: 'Llama 3.1 70B', paramsB: 70.6, layers: 80, kvHeads: 8, headDim: 128 },
  { id: 'custom', name: '직접 입력', paramsB: null, layers: null, kvHeads: null, headDim: null },
];

export function weightBytes({ paramsB, bits }) {
  return (paramsB * 1e9 * bits) / 8;
}

// KV cache는 FP16(2바이트) 기준. 아키텍처 정보가 없으면 8B 모델 대비 sqrt 스케일로 근사.
export function kvBytesPerToken({ paramsB, layers, kvHeads, headDim }) {
  if (layers && kvHeads && headDim) return 2 * layers * kvHeads * headDim * 2;
  return 131072 * Math.sqrt(paramsB / 8);
}

export function vramUsage(input) {
  const weights = weightBytes(input);
  const kv = kvBytesPerToken(input) * (input.promptTokens + input.outputTokens) * input.batch;
  const overhead = (weights + kv) * VRAM_OVERHEAD_RATIO;
  return { weights, kv, overhead, total: weights + kv + overhead };
}

// prefill: 연산량(compute-bound)과 가중치 1회 읽기(memory-bound) 중 큰 쪽 + 고정 오버헤드
export function prefillMs(input) {
  const flops = 2 * input.paramsB * 1e9 * input.promptTokens * input.batch;
  const computeMs = (flops / (input.tflops * 1e12 * input.gpuCount * COMPUTE_EFFICIENCY)) * 1000;
  const weightReadMs =
    (weightBytes(input) / (input.bandwidthGBs * 1e9 * input.gpuCount * MEM_EFFICIENCY)) * 1000;
  return Math.max(computeMs, weightReadMs) + PREFILL_OVERHEAD_MS;
}

// decode: 토큰 하나당 가중치 + 현재 KV cache 전체를 읽는다 (memory-bound)
export function tokenIntervalMs(input, contextTokens) {
  const bytes = weightBytes(input) + kvBytesPerToken(input) * contextTokens * input.batch;
  return (bytes / (input.bandwidthGBs * 1e9 * input.gpuCount * MEM_EFFICIENCY)) * 1000;
}

export function validate(input) {
  const errors = {};
  const positive = (k, label) => {
    if (!Number.isFinite(input[k]) || input[k] <= 0) errors[k] = `${label}은(는) 0보다 큰 숫자여야 합니다`;
  };
  const positiveInt = (k, label) => {
    if (!Number.isInteger(input[k]) || input[k] < 1) errors[k] = `${label}은(는) 1 이상의 정수여야 합니다`;
  };
  positive('vramGB', 'VRAM');
  positive('bandwidthGBs', '메모리 대역폭');
  positive('tflops', '연산 성능');
  positive('paramsB', '모델 크기');
  positiveInt('gpuCount', 'GPU 개수');
  positiveInt('promptTokens', '프롬프트 길이');
  positiveInt('outputTokens', '출력 길이');
  positiveInt('batch', '배치 크기');
  if (![4, 8, 16].includes(input.bits)) errors.bits = '양자화는 4, 8, 16 중 하나여야 합니다';
  return errors;
}

export function simulate(input) {
  const usage = vramUsage(input);
  const capacityBytes = input.vramGB * input.gpuCount * GIB;
  const startMs = tokenIntervalMs(input, input.promptTokens);
  const endMs = tokenIntervalMs(input, input.promptTokens + input.outputTokens);
  const tpsPerUser = 1000 / startMs;
  return {
    usage,
    capacityBytes,
    oom: usage.total > capacityBytes,
    ttftMs: prefillMs(input),
    tpsPerUser,
    tpsPerUserEnd: 1000 / endMs,
    tpsTotal: tpsPerUser * input.batch,
  };
}
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd /home/smallcloud/work/git/tps_sim && npm test`
Expected: PASS (모든 테스트 통과, fail 0)

- [ ] **Step 6: 커밋**

```bash
cd /home/smallcloud/work/git/tps_sim
git add package.json sim.js tests/sim.test.js
git commit -m "feat: add roofline-based LLM inference simulation model

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: UI (`index.html`, `style.css`, `app.js`)

**Files:**
- Create: `index.html`
- Create: `style.css`
- Create: `app.js`

**Interfaces:**
- Consumes (Task 1의 `sim.js`): `GPU_PRESETS`, `MODEL_PRESETS`, `validate(input)`, `simulate(input)`, `tokenIntervalMs(input, contextTokens)` — 시그니처는 Task 1 Interfaces 참고
- Produces: 없음 (최종 페이지)

- [ ] **Step 1: `index.html` 작성**

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM TPS 시뮬레이터</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main>
    <h1>LLM TPS 시뮬레이터</h1>
    <p class="sub">GPU와 모델 설정으로 첫 토큰 지연(TTFT)과 초당 토큰 수(TPS)를 추정합니다.</p>

    <div class="layout">
      <form id="form" class="panel" autocomplete="off" onsubmit="return false">
        <h2>하드웨어</h2>
        <label>GPU 프리셋<select id="gpuPreset"></select></label>
        <label>VRAM (GB / GPU)<input id="vramGB" type="number" step="any"><span class="err" id="err-vramGB"></span></label>
        <label>메모리 대역폭 (GB/s)<input id="bandwidthGBs" type="number" step="any"><span class="err" id="err-bandwidthGBs"></span></label>
        <label>FP16 연산 성능 (TFLOPS)<input id="tflops" type="number" step="any"><span class="err" id="err-tflops"></span></label>
        <label>GPU 개수<input id="gpuCount" type="number" value="1"><span class="err" id="err-gpuCount"></span></label>

        <h2>모델</h2>
        <label>모델 프리셋<select id="modelPreset"></select></label>
        <label>모델 크기 (B 파라미터)<input id="paramsB" type="number" step="any"><span class="err" id="err-paramsB"></span></label>
        <label>양자화
          <select id="bits">
            <option value="16">FP16 (16bit)</option>
            <option value="8">INT8 (8bit)</option>
            <option value="4">INT4 (4bit)</option>
          </select>
        </label>

        <h2>워크로드</h2>
        <label>프롬프트 길이 (토큰)<input id="promptTokens" type="number" value="1000"><span class="err" id="err-promptTokens"></span></label>
        <label>출력 길이 (토큰)<input id="outputTokens" type="number" value="300"><span class="err" id="err-outputTokens"></span></label>
        <label>배치 크기 (동시 사용자)<input id="batch" type="number" value="1"><span class="err" id="err-batch"></span></label>
      </form>

      <section class="panel">
        <h2>결과</h2>
        <div class="metrics">
          <div class="metric"><span class="label">TTFT</span><strong id="ttft">-</strong></div>
          <div class="metric"><span class="label">사용자당 TPS</span><strong id="tpsUser">-</strong></div>
          <div class="metric"><span class="label">총 처리량 TPS</span><strong id="tpsTotal">-</strong></div>
        </div>

        <div class="vram">
          <div class="vram-head"><span>VRAM 사용량</span><span id="vramText">-</span></div>
          <div class="bar"><div id="vramBar"></div></div>
        </div>
        <div id="warn" class="warn" hidden></div>

        <h2>출력 시뮬레이션</h2>
        <div class="controls">
          <button id="runBtn" type="button">▶ 실행</button>
          <button id="stopBtn" type="button" disabled>■ 정지</button>
          <label class="inline">속도
            <select id="speed">
              <option value="1">1x</option>
              <option value="5">5x</option>
              <option value="20">20x</option>
            </select>
          </label>
        </div>
        <div id="status" class="status">대기 중</div>
        <div id="stream" class="stream" aria-live="off"></div>

        <p class="note">이론적 추정치이며 실제 추론 엔진에서는 보통 이보다 낮게 나옵니다. GPU 여러 개는 이상적인 선형 확장을 가정합니다.</p>
      </section>
    </div>
  </main>
  <script type="module" src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: `style.css` 작성**

```css
:root {
  --bg: #0f1216; --panel: #181d24; --border: #2a313b; --text: #e6e9ee;
  --muted: #8b95a3; --accent: #4cc38a; --warn: #f0a04b; --danger: #e5484d;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", "Noto Sans KR", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { margin: 0; font-size: 26px; }
h2 { margin: 18px 0 8px; font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
h2:first-child { margin-top: 0; }
.sub { color: var(--muted); margin: 4px 0 20px; }
.layout { display: grid; grid-template-columns: 340px 1fr; gap: 16px; align-items: start; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
label { display: block; margin-bottom: 10px; font-size: 13px; color: var(--muted); }
label.inline { display: inline-flex; align-items: center; gap: 8px; margin: 0; }
input, select { display: block; width: 100%; margin-top: 4px; padding: 7px 9px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 6px; color: var(--text); font: inherit; }
label.inline select { width: auto; margin: 0; }
input.invalid { border-color: var(--danger); }
.err { display: block; color: var(--danger); font-size: 12px; min-height: 0; }
.metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.metric { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.metric .label { display: block; color: var(--muted); font-size: 12px; }
.metric strong { font-size: 22px; font-variant-numeric: tabular-nums; }
.vram { margin-top: 14px; }
.vram-head { display: flex; justify-content: space-between; font-size: 13px; color: var(--muted); margin-bottom: 4px; }
.bar { height: 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
#vramBar { height: 100%; width: 0; background: var(--accent); transition: width .2s; }
#vramBar.over { background: var(--danger); }
.warn { margin-top: 10px; padding: 8px 10px; border-radius: 6px; background: rgba(229,72,77,.12);
  border: 1px solid var(--danger); color: #ffb4b6; font-size: 13px; }
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
button { padding: 7px 14px; border-radius: 6px; border: 1px solid var(--border); background: var(--accent);
  color: #06210f; font: inherit; font-weight: 600; cursor: pointer; }
button:disabled { opacity: .45; cursor: not-allowed; }
#stopBtn { background: var(--bg); color: var(--text); }
.status { margin: 10px 0 6px; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
.stream { min-height: 120px; max-height: 260px; overflow: auto; padding: 10px 12px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 8px; font-family: ui-monospace, Menlo, Consolas, monospace;
  font-size: 13px; white-space: pre-wrap; word-break: break-word; }
.note { margin: 14px 0 0; font-size: 12px; color: var(--muted); }
@media (max-width: 800px) {
  .layout { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: 1fr; }
}
```

- [ ] **Step 3: `app.js` 작성**

```js
import { GPU_PRESETS, MODEL_PRESETS, validate, simulate, tokenIntervalMs } from './sim.js';

const $ = (id) => document.getElementById(id);
const NUM_FIELDS = ['vramGB', 'gpuCount', 'bandwidthGBs', 'tflops', 'paramsB', 'promptTokens', 'outputTokens', 'batch'];
const GB = 2 ** 30;
const WORDS = 'the of and to a in is that for it as with was on be by this are at from or an have not but'.split(' ');
const MAX_SHOWN_TOKENS = 400;

let raf = null;

function fillSelect(sel, items) {
  sel.innerHTML = items.map((i) => `<option value="${i.id}">${i.name}</option>`).join('');
}

function readInput() {
  const input = {};
  for (const id of NUM_FIELDS) input[id] = $(id).value === '' ? NaN : Number($(id).value);
  input.bits = Number($('bits').value);
  const model = MODEL_PRESETS.find((m) => m.id === $('modelPreset').value);
  if (model && model.layers) {
    input.layers = model.layers;
    input.kvHeads = model.kvHeads;
    input.headDim = model.headDim;
  }
  return input;
}

const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`);
const fmtTps = (v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));

function render() {
  const input = readInput();
  const errors = validate(input);
  for (const id of NUM_FIELDS) {
    $(`err-${id}`).textContent = errors[id] || '';
    $(id).classList.toggle('invalid', Boolean(errors[id]));
  }
  const valid = Object.keys(errors).length === 0;
  $('runBtn').disabled = !valid;
  if (!valid) {
    stop('입력 오류를 수정하세요');
    for (const id of ['ttft', 'tpsUser', 'tpsTotal', 'vramText']) $(id).textContent = '-';
    $('vramBar').style.width = '0';
    $('warn').hidden = true;
    return null;
  }
  const r = simulate(input);
  $('ttft').textContent = fmtMs(r.ttftMs);
  $('tpsUser').textContent = `${fmtTps(r.tpsPerUser)} tok/s`;
  $('tpsTotal').textContent = `${fmtTps(r.tpsTotal)} tok/s`;
  const used = r.usage.total / GB;
  const cap = r.capacityBytes / GB;
  $('vramText').textContent = `${used.toFixed(1)} / ${cap.toFixed(0)} GB`;
  $('vramBar').style.width = `${Math.min(100, (used / cap) * 100)}%`;
  $('vramBar').classList.toggle('over', r.oom);
  $('warn').hidden = !r.oom;
  if (r.oom) {
    $('warn').textContent =
      `VRAM 부족(OOM): 가중치 ${(r.usage.weights / GB).toFixed(1)} GB + KV cache ${(r.usage.kv / GB).toFixed(1)} GB ` +
      `+ 오버헤드가 ${cap.toFixed(0)} GB를 넘습니다. 양자화·배치·컨텍스트를 줄이거나 GPU를 늘리세요.`;
  }
  return { input, r };
}

function stop(message) {
  if (raf !== null) cancelAnimationFrame(raf);
  raf = null;
  $('stopBtn').disabled = true;
  if (message) $('status').textContent = message;
}

function run() {
  stop();
  const state = render();
  if (!state) return;
  const { input, r } = state;
  if (r.oom) {
    $('status').textContent = 'VRAM 부족으로 실행할 수 없습니다';
    return;
  }
  const speed = Number($('speed').value);
  const tokens = [];
  const stream = $('stream');
  stream.textContent = '';
  $('stopBtn').disabled = false;
  $('runBtn').disabled = true;

  const t0 = performance.now();
  let nextAt = r.ttftMs; // 가상 시간(ms): 첫 토큰이 나오는 시각
  let emitted = 0;

  function frame(now) {
    const virtual = (now - t0) * speed;
    if (virtual < r.ttftMs) {
      $('status').textContent = `Prefill 중… (${fmtMs(virtual)} / TTFT ${fmtMs(r.ttftMs)})`;
    } else {
      while (emitted < input.outputTokens && virtual >= nextAt) {
        tokens.push(WORDS[(emitted * 7 + 3) % WORDS.length]);
        nextAt += tokenIntervalMs(input, input.promptTokens + emitted);
        emitted++;
      }
      stream.textContent = tokens.slice(-MAX_SHOWN_TOKENS).join(' ');
      stream.scrollTop = stream.scrollHeight;
      const live = emitted / Math.max((virtual - r.ttftMs) / 1000, 1e-6);
      $('status').textContent = `생성 중 ${emitted} / ${input.outputTokens} 토큰 · 실측 ${fmtTps(live)} tok/s`;
    }
    if (emitted < input.outputTokens) {
      raf = requestAnimationFrame(frame);
    } else {
      raf = null;
      $('status').textContent = `완료 · ${input.outputTokens} 토큰 · 총 ${fmtMs(nextAt)} (가상 시간)`;
      $('stopBtn').disabled = true;
      $('runBtn').disabled = false;
    }
  }
  raf = requestAnimationFrame(frame);
}

function applyGpuPreset() {
  const p = GPU_PRESETS.find((g) => g.id === $('gpuPreset').value);
  if (!p || p.id === 'custom') return;
  $('vramGB').value = p.vramGB;
  $('bandwidthGBs').value = p.bandwidthGBs;
  $('tflops').value = p.tflops;
}

function applyModelPreset() {
  const m = MODEL_PRESETS.find((x) => x.id === $('modelPreset').value);
  if (!m || m.id === 'custom') return;
  $('paramsB').value = m.paramsB;
}

fillSelect($('gpuPreset'), GPU_PRESETS);
fillSelect($('modelPreset'), MODEL_PRESETS);
$('gpuPreset').value = 'rtx4090';
$('modelPreset').value = 'llama31-8b';
applyGpuPreset();
applyModelPreset();

$('gpuPreset').addEventListener('change', () => { applyGpuPreset(); render(); });
$('modelPreset').addEventListener('change', () => { applyModelPreset(); render(); });
for (const id of ['vramGB', 'bandwidthGBs', 'tflops']) {
  $(id).addEventListener('input', () => { $('gpuPreset').value = 'custom'; render(); });
}
$('paramsB').addEventListener('input', () => { $('modelPreset').value = 'custom'; render(); });
for (const id of ['gpuCount', 'promptTokens', 'outputTokens', 'batch', 'bits']) {
  $(id).addEventListener('input', render);
}
$('runBtn').addEventListener('click', run);
$('stopBtn').addEventListener('click', () => { stop('정지됨'); $('runBtn').disabled = false; });

render();
```

- [ ] **Step 4: 로컬 서버로 브라우저 동작 확인**

Run: `cd /home/smallcloud/work/git/tps_sim && python3 -m http.server 8765 &`

Playwright MCP로 `http://localhost:8765/` 를 열어 다음을 확인한다(ES module이라 `file://`은 안 됨).
1. 초기 화면: RTX 4090 + Llama 3.1 8B + FP16 → TTFT 약 `214 ms`(1000토큰), 사용자당 TPS 약 `49~50 tok/s`, VRAM 바 정상, 콘솔 에러 없음
2. 모델을 `Llama 3.1 70B`로 바꾸면 빨간 OOM 경고가 뜨고 실행 버튼을 눌러도 애니메이션이 시작되지 않음
3. 양자화를 INT4로 바꾸면 TPS가 4배 가까이 오르고, 70B는 GPU 개수를 4로 올리면 OOM이 해소됨
4. `VRAM`에 `0` 입력 → 필드에 오류 문구와 빨간 테두리, 실행 버튼 비활성화
5. 8B 기본값에서 속도 `20x`로 실행 → 토큰이 출력되고 상태가 "완료"로 바뀜. 실행 중 정지 버튼이 동작함
6. 창 폭 600px에서 가로 스크롤 없이 1열 배치

문제가 있으면 코드를 고치고 다시 확인한다. 확인 후 서버를 종료한다: `kill %1`

- [ ] **Step 5: 테스트 재실행 및 커밋**

Run: `cd /home/smallcloud/work/git/tps_sim && npm test`
Expected: PASS

```bash
git add index.html style.css app.js
git commit -m "feat: add simulator UI with token streaming animation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: 푸시 및 GitHub Pages 배포

**Files:** 없음 (배포 작업). 사용자가 푸시와 Pages 활성화를 이미 승인했다.

- [ ] **Step 1: 푸시**

Run: `cd /home/smallcloud/work/git/tps_sim && git push -u origin main`
Expected: `main -> main` 성공

- [ ] **Step 2: Pages 활성화 (main 브랜치 루트)**

Run: `gh api -X POST repos/bluerose7112/tps_sim/pages -f 'source[branch]=main' -f 'source[path]=/'`
Expected: JSON 응답에 `html_url`. 이미 활성화되어 있어 409/422가 나오면 `gh api repos/bluerose7112/tps_sim/pages` 로 상태를 확인한다.

- [ ] **Step 3: 배포 완료까지 대기 후 확인**

Run: `until [ "$(curl -s -o /dev/null -w '%{http_code}' https://bluerose7112.github.io/tps_sim/)" = "200" ]; do sleep 10; done; echo ready`
(최대 약 5분. 계속 200이 안 되면 `gh api repos/bluerose7112/tps_sim/pages/builds/latest` 로 빌드 상태를 확인한다.)

Playwright MCP로 https://bluerose7112.github.io/tps_sim/ 를 열어 결과 패널에 수치가 표시되고 콘솔 에러(특히 `sim.js`/`app.js` 404)가 없는지 확인한다.

- [ ] **Step 4: 결과 보고**

배포 URL과 확인한 내용(수치, 콘솔 에러 여부)을 사용자에게 보고한다.
