// LLM 추론 속도 계산 모델 (roofline 기반). DOM 의존 없음.

export const MEM_EFFICIENCY = 0.8;
export const COMPUTE_EFFICIENCY = 0.5;
export const PREFILL_OVERHEAD_MS = 20;
export const VRAM_OVERHEAD_RATIO = 0.1;
const GIB = 2 ** 30;

// 대역폭: GB/s, tflops: FP16 dense Tensor TFLOPS (FP32 누산 기준, 공개 사양에서 환산한 근사치)
// Apple Silicon은 통합메모리 용량을 vramGB에 넣었고 tflops는 GPU FP16 성능 근사치. 실제 GPU 사용 가능 메모리는 보통 전체의 약 75%.
export const GPU_PRESETS = [
  { id: 'rtx3060', name: 'RTX 3060 (12GB)', vramGB: 12, bandwidthGBs: 360, tflops: 25 },
  { id: 'rtx3060ti', name: 'RTX 3060 Ti (8GB)', vramGB: 8, bandwidthGBs: 448, tflops: 32 },
  { id: 'rtx3070', name: 'RTX 3070 (8GB)', vramGB: 8, bandwidthGBs: 448, tflops: 40 },
  { id: 'rtx3080', name: 'RTX 3080 (10GB)', vramGB: 10, bandwidthGBs: 760, tflops: 60 },
  { id: 'rtx3080ti', name: 'RTX 3080 Ti (12GB)', vramGB: 12, bandwidthGBs: 912, tflops: 68 },
  { id: 'rtx3090', name: 'RTX 3090 (24GB)', vramGB: 24, bandwidthGBs: 936, tflops: 71 },
  { id: 'rtx4060', name: 'RTX 4060 (8GB)', vramGB: 8, bandwidthGBs: 272, tflops: 30 },
  { id: 'rtx4060ti16', name: 'RTX 4060 Ti (16GB)', vramGB: 16, bandwidthGBs: 288, tflops: 44 },
  { id: 'rtx4070', name: 'RTX 4070 (12GB)', vramGB: 12, bandwidthGBs: 504, tflops: 58 },
  { id: 'rtx4070s', name: 'RTX 4070 Super (12GB)', vramGB: 12, bandwidthGBs: 504, tflops: 71 },
  { id: 'rtx4070ti', name: 'RTX 4070 Ti (12GB)', vramGB: 12, bandwidthGBs: 504, tflops: 80 },
  { id: 'rtx4070tis', name: 'RTX 4070 Ti Super (16GB)', vramGB: 16, bandwidthGBs: 672, tflops: 88 },
  { id: 'rtx4080', name: 'RTX 4080 (16GB)', vramGB: 16, bandwidthGBs: 717, tflops: 97 },
  { id: 'rtx4080s', name: 'RTX 4080 Super (16GB)', vramGB: 16, bandwidthGBs: 736, tflops: 104 },
  { id: 'rtx4090', name: 'RTX 4090 (24GB)', vramGB: 24, bandwidthGBs: 1008, tflops: 165 },
  { id: 'rtx5060', name: 'RTX 5060 (8GB)', vramGB: 8, bandwidthGBs: 448, tflops: 38 },
  { id: 'rtx5060ti', name: 'RTX 5060 Ti (16GB)', vramGB: 16, bandwidthGBs: 448, tflops: 47 },
  { id: 'rtx5070', name: 'RTX 5070 (12GB)', vramGB: 12, bandwidthGBs: 672, tflops: 62 },
  { id: 'rtx5070ti', name: 'RTX 5070 Ti (16GB)', vramGB: 16, bandwidthGBs: 896, tflops: 88 },
  { id: 'rtx5080', name: 'RTX 5080 (16GB)', vramGB: 16, bandwidthGBs: 960, tflops: 113 },
  { id: 'rtx5090', name: 'RTX 5090 (32GB)', vramGB: 32, bandwidthGBs: 1792, tflops: 209 },
  { id: 't4', name: 'Tesla T4 (16GB)', vramGB: 16, bandwidthGBs: 320, tflops: 65 },
  { id: 'v100', name: 'V100 (32GB)', vramGB: 32, bandwidthGBs: 900, tflops: 125 },
  { id: 'a10', name: 'A10 (24GB)', vramGB: 24, bandwidthGBs: 600, tflops: 125 },
  { id: 'l4', name: 'L4 (24GB)', vramGB: 24, bandwidthGBs: 300, tflops: 121 },
  { id: 'rtxa6000', name: 'RTX A6000 (48GB)', vramGB: 48, bandwidthGBs: 768, tflops: 77 },
  { id: 'rtx6000ada', name: 'RTX 6000 Ada (48GB)', vramGB: 48, bandwidthGBs: 960, tflops: 182 },
  { id: 'rtxpro6000', name: 'RTX PRO 6000 Blackwell (96GB)', vramGB: 96, bandwidthGBs: 1792, tflops: 250 },
  { id: 'l40s', name: 'L40S (48GB)', vramGB: 48, bandwidthGBs: 864, tflops: 362 },
  { id: 'a100-40', name: 'A100 (40GB)', vramGB: 40, bandwidthGBs: 1555, tflops: 312 },
  { id: 'a100', name: 'A100 (80GB)', vramGB: 80, bandwidthGBs: 2039, tflops: 312 },
  { id: 'h100', name: 'H100 SXM (80GB)', vramGB: 80, bandwidthGBs: 3350, tflops: 989 },
  { id: 'h200', name: 'H200 (141GB)', vramGB: 141, bandwidthGBs: 4800, tflops: 989 },
  { id: 'b200', name: 'B200 (192GB)', vramGB: 192, bandwidthGBs: 8000, tflops: 2250 },
  { id: 'rx7800xt', name: 'AMD RX 7800 XT (16GB)', vramGB: 16, bandwidthGBs: 624, tflops: 74 },
  { id: 'rx7900xt', name: 'AMD RX 7900 XT (20GB)', vramGB: 20, bandwidthGBs: 800, tflops: 103 },
  { id: 'rx7900xtx', name: 'AMD RX 7900 XTX (24GB)', vramGB: 24, bandwidthGBs: 960, tflops: 123 },
  { id: 'rx9070xt', name: 'AMD RX 9070 XT (16GB)', vramGB: 16, bandwidthGBs: 640, tflops: 195 },
  { id: 'w7900', name: 'AMD Radeon PRO W7900 (48GB)', vramGB: 48, bandwidthGBs: 864, tflops: 123 },
  { id: 'mi300x', name: 'AMD MI300X (192GB)', vramGB: 192, bandwidthGBs: 5300, tflops: 1307 },
  { id: 'mac-mini-m4-16', name: 'Mac mini M4 (16GB 통합메모리)', vramGB: 16, bandwidthGBs: 120, tflops: 4.6 },
  { id: 'mac-mini-m4-32', name: 'Mac mini M4 (32GB 통합메모리)', vramGB: 32, bandwidthGBs: 120, tflops: 4.6 },
  { id: 'mac-mini-m4pro-24', name: 'Mac mini M4 Pro (24GB 통합메모리)', vramGB: 24, bandwidthGBs: 273, tflops: 9.2 },
  { id: 'mac-mini-m4pro-48', name: 'Mac mini M4 Pro (48GB 통합메모리)', vramGB: 48, bandwidthGBs: 273, tflops: 9.2 },
  { id: 'mac-mini-m4pro-64', name: 'Mac mini M4 Pro (64GB 통합메모리)', vramGB: 64, bandwidthGBs: 273, tflops: 9.2 },
  { id: 'mac-studio-m4max-36', name: 'Mac Studio M4 Max 32코어 (36GB 통합메모리)', vramGB: 36, bandwidthGBs: 410, tflops: 14.7 },
  { id: 'mac-studio-m4max-64', name: 'Mac Studio M4 Max 40코어 (64GB 통합메모리)', vramGB: 64, bandwidthGBs: 546, tflops: 18.4 },
  { id: 'mac-studio-m4max-128', name: 'Mac Studio M4 Max 40코어 (128GB 통합메모리)', vramGB: 128, bandwidthGBs: 546, tflops: 18.4 },
  { id: 'mac-studio-m3ultra-96', name: 'Mac Studio M3 Ultra 60코어 (96GB 통합메모리)', vramGB: 96, bandwidthGBs: 819, tflops: 21 },
  { id: 'mac-studio-m3ultra-256', name: 'Mac Studio M3 Ultra 80코어 (256GB 통합메모리)', vramGB: 256, bandwidthGBs: 819, tflops: 28 },
  { id: 'mac-studio-m3ultra-512', name: 'Mac Studio M3 Ultra 80코어 (512GB 통합메모리)', vramGB: 512, bandwidthGBs: 819, tflops: 28 },
  { id: 'mac-studio-m2ultra-192', name: 'Mac Studio M2 Ultra (192GB 통합메모리)', vramGB: 192, bandwidthGBs: 800, tflops: 27 },
  { id: 'custom', name: '직접 입력', vramGB: null, bandwidthGBs: null, tflops: null },
];

// MoE 모델은 activeParamsB(토큰당 활성 파라미터)를 가진다. VRAM에는 전체 파라미터가 올라가고 decode는 활성 가중치만 읽는다.
// Gemma 3의 sliding window는 무시(KV 과대 추정). DeepSeek은 MLA라 latent 캐시(576차원/레이어)를 kvHeads=1, headDim=288로 근사.
export const MODEL_PRESETS = [
  { id: 'llama32-3b', name: 'Llama 3.2 3B', paramsB: 3.21, layers: 28, kvHeads: 8, headDim: 128 },
  { id: 'qwen3-4b', name: 'Qwen3 4B', paramsB: 4.0, layers: 36, kvHeads: 8, headDim: 128 },
  { id: 'qwen25-7b', name: 'Qwen2.5 7B', paramsB: 7.6, layers: 28, kvHeads: 4, headDim: 128 },
  { id: 'mistral-7b', name: 'Mistral 7B', paramsB: 7.24, layers: 32, kvHeads: 8, headDim: 128 },
  { id: 'llama31-8b', name: 'Llama 3.1 8B', paramsB: 8.03, layers: 32, kvHeads: 8, headDim: 128 },
  { id: 'qwen3-8b', name: 'Qwen3 8B', paramsB: 8.2, layers: 36, kvHeads: 8, headDim: 128 },
  { id: 'gemma3-4b', name: 'Gemma 3 4B', paramsB: 4.3, layers: 34, kvHeads: 4, headDim: 256 },
  { id: 'gemma3-12b', name: 'Gemma 3 12B', paramsB: 12.2, layers: 48, kvHeads: 8, headDim: 256 },
  { id: 'mistral-nemo-12b', name: 'Mistral Nemo 12B', paramsB: 12.2, layers: 40, kvHeads: 8, headDim: 128 },
  { id: 'qwen3-14b', name: 'Qwen3 14B', paramsB: 14.8, layers: 40, kvHeads: 8, headDim: 128 },
  { id: 'phi4-14b', name: 'Phi-4 14B', paramsB: 14.7, layers: 40, kvHeads: 10, headDim: 128 },
  { id: 'mistral-small-24b', name: 'Mistral Small 3 24B', paramsB: 23.6, layers: 40, kvHeads: 8, headDim: 128 },
  { id: 'gemma3-27b', name: 'Gemma 3 27B', paramsB: 27.4, layers: 62, kvHeads: 16, headDim: 128 },
  { id: 'qwen25-32b', name: 'Qwen2.5 32B', paramsB: 32.5, layers: 64, kvHeads: 8, headDim: 128 },
  { id: 'qwen3-32b', name: 'Qwen3 32B', paramsB: 32.8, layers: 64, kvHeads: 8, headDim: 128 },
  { id: 'llama31-70b', name: 'Llama 3.1 / 3.3 70B', paramsB: 70.6, layers: 80, kvHeads: 8, headDim: 128 },
  { id: 'llama31-405b', name: 'Llama 3.1 405B', paramsB: 405, layers: 126, kvHeads: 8, headDim: 128 },
  { id: 'gptoss-20b', name: 'gpt-oss 20B (MoE, 활성 3.6B)', paramsB: 21, activeParamsB: 3.6, layers: 24, kvHeads: 8, headDim: 64 },
  { id: 'qwen3-30b-a3b', name: 'Qwen3 30B-A3B (MoE, 활성 3.3B)', paramsB: 30.5, activeParamsB: 3.3, layers: 48, kvHeads: 4, headDim: 128 },
  { id: 'gptoss-120b', name: 'gpt-oss 120B (MoE, 활성 5.1B)', paramsB: 117, activeParamsB: 5.1, layers: 36, kvHeads: 8, headDim: 64 },
  { id: 'llama4-scout', name: 'Llama 4 Scout (MoE, 활성 17B)', paramsB: 109, activeParamsB: 17, layers: 48, kvHeads: 8, headDim: 128 },
  { id: 'qwen3-235b-a22b', name: 'Qwen3 235B-A22B (MoE, 활성 22B)', paramsB: 235, activeParamsB: 22, layers: 94, kvHeads: 4, headDim: 128 },
  { id: 'llama4-maverick', name: 'Llama 4 Maverick (MoE, 활성 17B)', paramsB: 400, activeParamsB: 17, layers: 48, kvHeads: 8, headDim: 128 },
  { id: 'deepseek-v3', name: 'DeepSeek V3 / R1 (MoE, 활성 37B)', paramsB: 671, activeParamsB: 37, layers: 61, kvHeads: 1, headDim: 288 },
  { id: 'gemma4-12b', name: 'Gemma 4 12B', paramsB: 12, layers: 48, kvHeads: 8, headDim: 256 },
  { id: 'gemma4-31b', name: 'Gemma 4 31B', paramsB: 31, layers: 64, kvHeads: 4, headDim: 256 },
  { id: 'gemma4-26b-a4b', name: 'Gemma 4 26B-A4B (MoE, 활성 4B)', paramsB: 26, activeParamsB: 4, layers: 30, kvHeads: 4, headDim: 256 },
  { id: 'glm53-flash', name: 'GLM-5.3-Flash (MoE, 활성 18B)', paramsB: 320, activeParamsB: 18, layers: 45, kvHeads: 64, headDim: 64 },
  { id: 'glm53', name: 'GLM-5.3 (MoE, 활성 40B)', paramsB: 745, activeParamsB: 40, layers: 78, kvHeads: 64, headDim: 64 },
  { id: 'kimi-k3', name: 'Kimi K3 (MoE, 활성 104B, MLA 근사)', paramsB: 2800, activeParamsB: 104, layers: 93, kvHeads: 1, headDim: 512 },
  { id: 'custom', name: '직접 입력', paramsB: null, layers: null, kvHeads: null, headDim: null },
];

export function weightBytes({ paramsB, bits }) {
  return (paramsB * 1e9 * bits) / 8;
}

// MoE: 토큰 하나가 읽는/계산하는 활성 파라미터 (없으면 전체)
export function activeWeightBytes(input) {
  return weightBytes({ paramsB: input.activeParamsB ?? input.paramsB, bits: input.bits });
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
  const flops = 2 * (input.activeParamsB ?? input.paramsB) * 1e9 * input.promptTokens * input.batch;
  const computeMs = (flops / (input.tflops * 1e12 * input.gpuCount * COMPUTE_EFFICIENCY)) * 1000;
  const weightReadMs =
    (weightBytes(input) / (input.bandwidthGBs * 1e9 * input.gpuCount * MEM_EFFICIENCY)) * 1000;
  return Math.max(computeMs, weightReadMs) + PREFILL_OVERHEAD_MS;
}

// decode: 토큰 하나당 가중치 + 현재 KV cache 전체를 읽는다 (memory-bound)
export function tokenIntervalMs(input, contextTokens) {
  const bytes = activeWeightBytes(input) + kvBytesPerToken(input) * contextTokens * input.batch;
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
