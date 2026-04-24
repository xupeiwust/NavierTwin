# NavierTwin

CFD 후처리 결과를 AI/ROM 기반 디지털 트윈으로 변환하는 로컬 GUI 도구입니다.

## 현재 상태

- 현재 개발 단계: `v1.1.0` (ROADMAP 기준)
- 패키지 버전: `0.1.0` (alpha)
- 핵심 구현: CFD I/O, 기초 유동분석, POD/Randomized POD, Surrogate, TwinEngine, GUI 뼈대

상세 범위는 아래 문서를 참고하세요.

- 기술 명세: [`SPEC.md`](SPEC.md)
- 구현 계획: [`PLAN.md`](PLAN.md)
- 진행 현황: [`ROADMAP.md`](ROADMAP.md)

## 빠른 시작 (개발 환경)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

빠른 테스트는 기본 의존성만으로 충분합니다.

```bash
pytest -q
```

GUI 또는 확장 기능이 필요하면 `core` 또는 `full` extras를 사용하세요.

```bash
pip install -e ".[core]"
```

```bash
pip install -e ".[full]"
```

GUI 실행:

```bash
python3 main.py --gui
```

또는 editable install 이후:

```bash
naviertwin --gui
```

## 프로젝트 구조

```text
src/naviertwin/core/    # reader, 분석, 축소, surrogate, twin engine
src/naviertwin/gui/     # 메인 윈도우, 6단계 패널, 위젯
tests/                  # 단위 테스트/픽스처
```

## 프로젝트 저장

- `.ntwin` 프로젝트 저장 시 `모델(TwinEngine) 포함`이 체크되면
  같은 경로에 sidecar 파일 `.engine.pkl`을 함께 저장합니다.

## 테스트 범위

- 빠른 검증: `pytest -q`
- CI 기본 검증: `pytest -q -m "not optional"`
- optional 의존성 필요 경로: `pyvista`, `gmsh`, `pycgns` 등은 `core` 또는 `full` extras와 함께 설치하세요.
