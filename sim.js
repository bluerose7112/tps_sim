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
