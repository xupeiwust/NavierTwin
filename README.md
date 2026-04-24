# NavierTwin

CFD 후처리 결과를 AI/ROM/Operator Learning 기반 디지털 트윈으로 변환하는 로컬 GUI 도구입니다.

## 현재 상태

- 개발 단계: **v4.2.0** (ROADMAP 기준)
- 패키지 버전: `4.2.0`
- 250+ 테스트 통과, ruff 린트 깨끗
- 로컬 실행 · GPL-3.0 오픈소스 · 데이터 외부 유출 없음

상세 범위는 다음 문서 참고.

- 기술 명세: [`SPEC.md`](SPEC.md)
- 구현 계획: [`PLAN.md`](PLAN.md)
- 진행 현황: [`ROADMAP.md`](ROADMAP.md)

## 구현된 핵심 기능

### CFD I/O
- OpenFOAM (pv/ofpp/foamlib 폴백), VTK, Fluent (.cas/.dat ASCII), CGNS, Gmsh (.msh), SU2 (.su2)
- 내부 포맷: `.ntwin` (VTKHDF 확장, HDF5)

### 전처리
- **메쉬 생성**: 파라미터 채널/원통/NACA 익형 (Gmsh OCC)
- **메쉬 후처리**: PyMeshLab 단순화/스무딩 + PyVista 품질 보고서

### 차원축소
- 선형: Snapshot POD, Randomized SVD, Constrained POD (제약 null-space 투영)
- 비선형: Autoencoder, β-VAE (샘플링 생성), GNN-AE (torch_geometric)
- 텐서: Tucker 분해 (HOSVD + HOOI)
- 기하학적: Diffusion Maps (Coifman-Lafon α-정규화)

### 모달/통계
- DMD, SPOD (Welch-block + PySPOD 옵션), PGD (3D greedy)
- FFT/PSD, CWT 웨이블릿, 두점 상관 + 적분 길이
- LCS FTLE (RK4 flow-map + Cauchy-Green)

### 유동 분석
- Q-criterion / λ₂, y+ / Cf / δ₉₉ / δ* / θ / H
- 무차원수: Re, Pr, Nu, Pe, Gr, Ra
- 엔트로피 생성율 (Bejan, thermal + viscous)
- Couette / Poiseuille 2D / Pipe 해석해 + 수치해 자동 비교

### Surrogate
- RBF, Kriging (SMT), Bayesian Optimization (GP + EI), Co-Kriging 멀티피델리티

### 신경 연산자
- **FNO1D/2D**, TFNO2D (Tucker-factorized), WNO1D (웨이블릿)
- **DeepONet**, PI-DeepONet (물리 잔차), MIONet (multi-input)
- U-Net 2D, KANO1D (KAN + spectral)
- C4 Equivariant FNO (회전 평균)

### GNN
- GNN Surrogate (GCN N-layer), MeshGraphNets (Encode-Process-Decode)

### 시계열 / Koopman
- LSTM, Temporal Transformer, Neural ODE (torchdiffeq + RK4 폴백)
- KNO (Koopman Neural Operator), Latent Dynamics (AE + Neural ODE)

### 생성 모델
- Score-based Diffusion PDE (DDPM 형태)

### PINN / 물리 보정
- PINNSolver (PDE 잔차 + BC), Hybrid ROM (POD + NN 잔차)
- 선형 제약 투영, 질량 보존 스케일링
- SINDy (STLSQ + PySINDy 옵션)

### 데이터 동화 / UQ / 최적화
- EnKF (inflation), Particle Filter (SIR + systematic resample), 4D-Var (선형)
- Sobol 민감도 (Saltelli + SALib 옵션), MC 전파, KernelSHAP
- NSGA-II 다목적, SIMP 위상 최적화 (2D), Bayesian Optimization

### 멀티피델리티 / Transfer / Active
- Additive Co-Kriging, freeze/finetune 전이학습
- Variance-based active learning + loop

### 디지털 트윈 / GUI
- `NavierTwinPipeline`: 6 단계 end-to-end 오케스트레이터
- GUI: Import / Analyze / Reduce / Model / Twin / Export 6 탭
- 분석 패널: Q-criterion / λ₂ / FFT / y+ / 해석해 비교 (Matplotlib 임베드)
- 모델 패널: Kriging/RBF + 신경 연산자 (FNO/TFNO/DeepONet/UNet/WNO)
- 모델 비교 대시보드, loss curve 위젯, 튜토리얼 위자드
- Undo/Redo Command 스택, i18n (ko/en)

### 설명가능성
- Kernel SHAP (MC Shapley), Granger causality, 상관행렬

### Export / 배포
- ONNX, TorchScript (trace/script)
- Jinja2 HTML 보고서 + weasyprint PDF
- FastAPI REST 서버 (`/health`, `/reduce/pod`, `/analytic/*`, `/optimize/bayesian`)
- PyInstaller spec + Inno Setup (`installer/naviertwin.iss`)

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[core,dev]"

# 테스트
pytest -q -m "not optional"

# GUI 실행
python3 main.py --gui
```

선택 의존성 (Gmsh / PyMeshLab / Dedalus / FastAPI / PyTorch Geometric):

```bash
pip install -e ".[full]"
```

## API 사용 예시

```python
from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 30))  # (n_features, n_snapshots)

pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=5, surrogate_kind="kriging")
pipe.load_snapshots(X, field_name="U")
pipe.reduce()
params = np.linspace(0, 1, 30).reshape(-1, 1)
pipe.fit_surrogate(params)
metrics = pipe.validate(params[-8:], pipe.state.coeffs[-8:])
pipe.export_report("report.html", project="Demo")
```

## REST 서버

```bash
uvicorn naviertwin.api.server:app --host 0.0.0.0 --port 8000
```

엔드포인트: `/health`, `/analytic/couette`, `/analytic/poiseuille_2d`, `/reduce/pod`, `/optimize/bayesian`.

## 프로젝트 구조

```text
src/naviertwin/
├── core/
│   ├── cfd_reader/           # OpenFOAM/VTK/Fluent/CGNS/Gmsh/SU2
│   ├── dimensionality_reduction/  # POD/rSVD/cPOD/AE/VAE/GNN-AE/Tucker/DiffMaps
│   ├── flow_analysis/        # Q-crit/DMD/SPOD/FFT/CWT/2pc/BL/nondim/LCS/PGD/엔트로피
│   ├── surrogate/            # RBF/Kriging
│   ├── operator_learning/    # FNO/TFNO/WNO/DeepONet/PI-DeepONet/MIONet/UNet/KANO/KNO
│   ├── gnn/                  # GNN Surrogate / MeshGraphNets
│   ├── time_series/          # LSTM/Transformer/NeuralODE/Latent Dynamics
│   ├── generative/           # Diffusion PDE
│   ├── equivariant/          # C4 Equivariant FNO
│   ├── physnemo/             # PINNSolver
│   ├── physics_correction/   # 제약 투영 + Hybrid ROM
│   ├── data_assimilation/    # EnKF/PF/4D-Var
│   ├── optimization/         # BO/NSGA-II/SIMP/MC
│   ├── sensitivity/          # Sobol/Granger/correlation
│   ├── multi_fidelity/       # Co-Kriging/Transfer
│   ├── online_learning/      # Active learning
│   ├── explainability/       # KernelSHAP
│   ├── digital_twin/         # TwinEngine / Pipeline
│   ├── validation/           # metrics / analytic solutions
│   ├── export/               # .ntwin / ONNX / TorchScript
│   ├── report/               # Jinja2 HTML/PDF
│   └── tools/                # Gmsh 메쉬 생성 / PyMeshLab 후처리
├── gui/
│   ├── panels/               # 6 탭 패널
│   ├── widgets/              # VTK viewer / loss curve / compare / analytic compare
│   ├── wizard/               # 튜토리얼
│   └── styles/               # QSS + i18n (ko/en)
├── api/                      # FastAPI REST 서버
└── utils/                    # config / logger / undo_redo / i18n
```

## 테스트 전략

- 빠른 검증: `pytest -q -m "not optional"`
- 전체 (optional 의존성 포함): `pytest -q`
- optional 의존성이 필요한 모듈은 자동 skip — `pymeshlab` · `dedalus` · `PyWavelets` · `pysindy` 등.

## 라이선스

GPL-3.0 (비상업용). 자세한 내용은 `LICENSE` 참조.
