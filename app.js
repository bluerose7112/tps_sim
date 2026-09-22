import { GPU_PRESETS, MODEL_PRESETS, validate, simulate, tokenIntervalMs } from './sim.js';

const $ = (id) => document.getElementById(id);
const NUM_FIELDS = ['vramGB', 'gpuCount', 'bandwidthGBs', 'tflops', 'paramsB', 'promptTokens', 'outputTokens', 'batch'];
const GB = 2 ** 30;
const WORDS = 'the of and to a in is that for it as with was on be by this are at from or an have not but'.split(' ');
const MAX_SHOWN_TOKENS = 400;

let raf = null;

function fillSelect(sel, items, label = (i) => i.name) {
  sel.innerHTML = items.map((i) => `<option value="${i.id}">${label(i)}</option>`).join('');
}

// 원본(FP16, 2바이트/파라미터) 가중치 용량 근사치. 실제 배포 파일은 양자화·메타데이터에 따라 다를 수 있음.
function modelSizeLabel(m) {
  if (!(m.paramsB > 0)) return m.name;
  const gb = m.paramsB * 2;
  const size = gb >= 1000 ? `${(gb / 1000).toFixed(1)} TB` : `${gb.toFixed(gb < 10 ? 1 : 0)} GB`;
  return `${m.name} · 원본(FP16) ~${size}`;
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
    if (model.activeParamsB) input.activeParamsB = model.activeParamsB;
  }
  return input;
}

const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`);
const fmtTps = (v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));

function render() {
  const input = readInput();
  if (raf === null) $('status').textContent = '대기 중';
  const errors = validate(input);
  for (const id of NUM_FIELDS) {
    $(`err-${id}`).textContent = errors[id] || '';
    $(id).classList.toggle('invalid', Boolean(errors[id]));
  }
  const valid = Object.keys(errors).length === 0;
  $('runBtn').disabled = !valid || raf !== null;
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
      $('status').textContent = `생성 중 ${emitted} / ${input.outputTokens} 토큰 · 환산 ${fmtTps(live)} tok/s`;
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
fillSelect($('modelPreset'), MODEL_PRESETS, modelSizeLabel);
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
