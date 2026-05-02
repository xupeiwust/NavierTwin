# NavierTwin GUI 검증 가이드

10개 탭 모두 직접 테스트하는 절차. 각 탭에서 **기본 워크플로 → 에지 케이스
→ 회귀 검증** 순으로 진행.

## 0. 실행

```bash
# 권장 (소스 체크아웃에서 직접 실행)
PYTHONPATH=src python3 -m naviertwin --gui

# 또는 설치 후
naviertwin --gui
```

언어 전환: 메뉴 `보기 (V)` → `English / 한국어`.

## 사전 준비 — 합성 데이터셋

테스트용 .ntwin 파일이 없으면 아래 명령으로 생성:

```bash
PYTHONPATH=src python3 -c "
from naviertwin.examples.cavity_benchmark import generate_cavity_dataset
import numpy as np
ds = generate_cavity_dataset(n_t=30, n_x=40, n_y=40)
ds.save('/tmp/cavity.ntwin')
print('saved /tmp/cavity.ntwin')
"
```

(예제 스크립트가 없으면 ① Import 탭의 `더미 데이터셋 생성` 기능 사용)

---

## 탭 ① Import (불러오기)

**목적**: CFD 결과 파일 (.foam / .cgns / .vtk / .vtu / .nc / .ntwin) 로드.

### 기본 워크플로
1. **`파일 선택`** → `/tmp/cavity.ntwin` 선택
2. **`Readiness 점검`** 버튼 → "OK" 메시지
3. **`데이터 로드`** → 우측 패널에 mesh 정보 (n_points, fields) 표시

### 검증 포인트
- ✅ 메쉬 정점 수 / 셀 수 표시
- ✅ field 목록 (예: U, p, T) 표시
- ✅ 우측 3D 뷰어에 mesh 렌더링 (PyVista)

### 에지 케이스
- 잘못된 파일 → "지원하지 않는 형식" 에러 메시지
- 빈 폴더 선택 → 안내 메시지

---

## 탭 ② Analyze (분석)

**목적**: Q-criterion / λ₂ / FFT-PSD / y+ / 해석해 비교.

### 기본 워크플로
1. ① Import에서 데이터 로드 후 ② Analyze로 이동
2. 좌측 메서드 리스트에서 **`Q-criterion`** 선택
3. 속도 필드: `U` 선택
4. **`분석 실행`** → 우측 결과 텍스트 영역에 통계 출력

### 검증 포인트
- ✅ Q-criterion: `Q-criterion` field가 mesh에 추가됨
- ✅ FFT/PSD: 주파수 피크 표시
- ✅ y+ 분석: 벽면 전단응력 → y+ 분포
- ✅ 해석해 비교: Couette / Poiseuille 정확도 ≥ 0.99

---

## 탭 ③ Reduce (차원 축소)

**목적**: POD / Randomized POD / Incremental POD / MRPOD / AE / VAE.

### 기본 워크플로
1. 좌측 reducer 선택 (예: **POD**)
2. 모드 수 입력 (예: 5)
3. **`축소 실행`** → 누적 에너지 / 잠재 공간 좌표 표시

### 검증 포인트
- ✅ POD 99% 에너지 보존: 모드 수 자동 결정
- ✅ Autoencoder/VAE: PyTorch 설치 필요 (없으면 disabled)
- ✅ MRPOD: scale별 에너지 분포

---

## 탭 ④ Model (모델 / 운영자 학습)

**목적**: Kriging / RBF / FNO1D / DeepONet 등 surrogate 학습.

### 기본 워크플로
1. **`모델 학습`** → 학습 진행률 + loss curve 위젯 갱신
2. **`후보 추천`**: Bayesian Optimization → 다음 평가점 제안
3. **`연산자 학습`**: FNO/DeepONet 학습 (PyTorch 필요)

### 검증 포인트
- ✅ 모델 비교 탭 ⑦ 자동 갱신 (RMSE/R²)
- ✅ Loss curve 실시간 업데이트
- ✅ 모델별 학습 시간 / 메트릭

---

## 탭 ⑤ Twin (디지털 트윈)

**목적**: 파이프라인 빌드 / 예측 / 최적화 / 동화.

### 기본 워크플로
1. **`예측 실행`** → 학습된 surrogate로 새 파라미터 예측
2. **`최적화 실행`**: Bayesian Opt 결과 plot
3. **`동화 quick-check`**: EnKF/UKF 데이터 동화
4. **`저장 / 로드`**: pipeline state HDF5

### 검증 포인트
- ✅ 예측이 실제와 비교 가능 (R²)
- ✅ 최적화 수렴 곡선
- ✅ HDF5 저장/복원 round-trip

---

## 탭 ⑥ Export (내보내기)

**목적**: VTK / CSV / ONNX / TorchScript / 보고서 PDF.

### 기본 워크플로
1. 출력 형식 선택 (VTK / CSV / ONNX / 보고서)
2. **`찾기`**로 저장 경로 선택
3. **`내보내기`** → 파일 생성

### 검증 포인트
- ✅ VTK: ParaView로 열림
- ✅ CSV: 엑셀로 열림
- ✅ ONNX: `onnx.checker.check_model` 통과
- ✅ 보고서 PDF: 그림 + 메트릭 표

---

## 탭 ⑦ Compare (모델 비교)

**목적**: 학습된 모델들의 RMSE / R² 바 차트.

### 검증 포인트
- ✅ ④ Model에서 학습 후 자동 갱신
- ✅ 테이블 + 차트 동시 표시 (수동 버튼 없음)

---

## 탭 ⑧ Simulation (시뮬레이션)

**목적**: LBM / Streaming / RL / Burgers FNO 라이브 데모.

### 기본 워크플로
1. 시뮬레이션 종류 선택
2. **`시뮬레이션 실행`** → 시간 진행 애니메이션

### 검증 포인트
- ✅ 실시간 frame 업데이트
- ✅ 정지 / 재시작 가능

---

## 탭 ⑨ Explain (설명가능성)

**목적**: SHAP / Symbolic regression / Attention 시각화.

### 기본 워크플로
1. **`SHAP 설명 실행`** → 입력 변수별 기여도
2. **`Symbolic 식 추정`** → ROM 계수의 해석적 식
3. **`Attention 시각화 실행`** → Transformer attention map

### 검증 포인트
- ✅ SHAP 막대 차트
- ✅ Symbolic 식 LaTeX 표시
- ✅ Attention heatmap

---

## 탭 ⑩ Post-Tools (후처리 도구) ⭐ R591-647 통합

**목적**: 52개 후처리 op (PSD / EOF / Reynolds stats / change points / ...) 통합 GUI.

### 기본 워크플로 (Demo 데이터)
1. 좌측 **카테고리** 콤보 → `spectral` 선택
2. 연산 리스트 → **`psd_welch`** 더블클릭
3. 우측 설명 패널에 op 정보 표시
4. **Scalar 파라미터**: `fs=1000`, `nperseg=512` 입력
5. **프리셋**: `high_resolution` 선택 (factory preset 자동 적용)
6. **`Demo 실행 (합성 데이터)`** → 차트 + 텍스트 결과

### 검증 포인트 — 핵심 op 5종

| op | 카테고리 | 차트 검증 |
|----|----------|-----------|
| `psd_welch` | spectral | log-log 곡선, 피크 주파수 ≈ 5Hz |
| `eof` | rom | 4개 모드 subplot, 첫 모드 에너지 최대 |
| `reynolds_stats` | statistics | mean/RMS/TKE 텍스트 |
| `change_points` | anomaly | 변화점 위치 마커 |
| `quadrant_analysis` | statistics | Q1-Q4 4-bar chart |

### 결과 export 검증
1. op 실행 후 **`CSV`** 클릭 → 저장 경로 선택 → 엑셀로 열림
2. **`JSON`** → 큰 배열은 summary, 작은 건 full
3. **`NPZ`** → `np.load()`로 복원
4. **`차트 이미지 저장`** → PNG / SVG / PDF 선택 가능

### 일괄 실행 검증
1. 카테고리: `statistics` 선택
2. **`카테고리 일괄 실행`** → 모든 statistics op 실행 → markdown 요약 (✅/❌)

### 사용자 프리셋 검증
1. `denoise` 선택, `window_length=21`, `polyorder=5` 입력
2. **`프리셋 저장`** → 이름 `my_smooth` 입력
3. 다른 op로 갔다가 다시 `denoise` 선택 → 프리셋 콤보에 `my_smooth` 보임

### 이력 검증
1. 여러 op 실행
2. **`이력 보기`** → 다이얼로그에 시간순 테이블 표시
3. 항목 더블클릭 → 패널이 op + 파라미터 자동 복원
4. **`재실행`** 버튼

### Dataset 연결 검증
1. ① Import에서 데이터 로드
2. ⑩ Post-Tools 좌측 상단 **데이터 라벨** "로드됨" 으로 변경
3. **필드 콤보** 활성화 → `U`, `p`, `T` 등 선택 가능
4. `psd_welch` 실행 → 입력 라벨 "로드 데이터셋"

### 언어 전환 검증
1. 메뉴 `View → English` → 모든 라벨 영어로 변경
2. `View → 한국어` → 한국어로 복귀
3. 카테고리/프리셋 콤보 첫 항목도 변경 (`전체` ↔ `All`)

---

## 자동화 회귀 검증 (CLI)

GUI 직접 클릭 대신 자동 smoke test로 검증:

```bash
# 모든 GUI 패널 + facade smoke
QT_QPA_PLATFORM=offscreen pytest tests/test_postproc_*.py tests/test_main_window_*.py -q

# Post-Tools 단독
QT_QPA_PLATFORM=offscreen pytest tests/test_postproc_panel.py tests/test_postproc_chart.py -q

# Facade (52개 op core 검증)
pytest tests/test_post_process_facade.py tests/test_facade_rom_extensions.py tests/test_facade_round653.py -q
```

기대: **130+ tests pass, 0 failed**.

---

## 알려진 제약

- **PyTorch 미설치**: ④ Model의 FNO/DeepONet, ③ Reduce의 AE/VAE 비활성
- **PyVista 미설치**: ① Import의 3D 뷰어 비활성 (mesh 정보만 텍스트)
- **pywt 미설치**: WNO / WaveletDiffusion 관련 op는 skip
- **gmsh 미설치**: 메쉬 생성 기능 (① Import의 더미) 제한
- **matplotlib 미설치**: ⑩ Post-Tools 차트 영역 미표시 (텍스트만)

위 의존성은 모두 **optional**. 핵심 후처리/ROM 기능은 numpy + scipy + PySide6만으로 동작.
