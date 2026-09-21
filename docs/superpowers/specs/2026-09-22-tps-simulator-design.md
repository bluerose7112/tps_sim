# LLM TPS Simulator 설계

## 목적
GPU 사양과 모델 설정을 입력하면 LLM 추론의 TTFT(첫 토큰까지의 지연)와 TPS(초당 토큰 수)를
roofline 모델로 계산하고, 토큰이 출력되는 모습을 애니메이션으로 보여주는 정적 웹페이지.
GitHub Pages(`main` 브랜치 루트)로 배포한다.

## 구조 (바닐라 HTML/JS/CSS, 빌드 없음)
- `index.html` — 입력 폼, 결과 패널, 애니메이션 영역
- `sim.js` — 순수 계산 함수 (DOM 의존 없음, Node에서 테스트 가능, ES module)
- `app.js` — 폼 바인딩, 결과 렌더링, 토큰 출력 애니메이션
- `style.css` — 스타일
- `tests/sim.test.js` — `node --test`로 실행하는 sim.js 단위 테스트

## 입력 인자
VRAM(GB, GPU당), GPU 프리셋(메모리 대역폭 GB/s, FP16 TFLOPS; 직접 입력 가능), GPU 개수,
모델 크기(파라미터 B), 양자화(FP16/INT8/INT4), 프롬프트 길이, 출력 길이, 배치 크기.

## 계산 모델
- 가중치 바이트 = 파라미터 수 × 비트/8
- KV cache 바이트 = 2 × 레이어 수 × KV 헤드 수 × 헤드 차원 × 컨텍스트 길이 × 배치 × 2바이트
  (모델 프리셋에서 레이어/KV헤드/헤드차원을 가져오고, 프리셋이 없으면 파라미터 수에서 근사)
- VRAM 사용량 = 가중치 + KV cache + 오버헤드(약 10%). 총 VRAM(VRAM × GPU 수)을 넘으면 OOM 경고
- TTFT = 프롬프트 토큰 × 2 × 파라미터 / (TFLOPS × GPU 수 × 효율) + 고정 오버헤드
- decode 토큰 시간 = (가중치 + 현재 KV cache 읽기량) / (대역폭 × GPU 수 × 효율)
- 사용자당 TPS = 1 / decode 토큰 시간, 총 처리량 = 사용자당 TPS × 배치
- 결과는 이론 상한에 가까운 추정치이므로 화면에 안내 문구를 표시한다

## 애니메이션
계산된 TTFT만큼 대기한 뒤, 사용자당 TPS 간격으로 토큰을 하나씩 출력한다.
실제 시간 그대로 재생하고 속도 배율(1x/5x/20x)을 선택할 수 있다. 다시 실행/정지 버튼을 제공한다.
OOM이면 애니메이션을 시작하지 않고 경고만 표시한다.

## 에러 처리
잘못된 입력(0 이하, 빈 값)은 계산을 건너뛰고 필드에 오류를 표시한다.

## 테스트
`sim.js`의 순수 함수(VRAM 계산, TTFT, TPS, OOM 판정)를 `node --test`로 검증한다.
UI는 브라우저(Playwright)로 실제 동작을 확인한다.

## 배포
`main` 브랜치 루트를 GitHub Pages 소스로 설정한다. URL: https://bluerose7112.github.io/tps_sim/
