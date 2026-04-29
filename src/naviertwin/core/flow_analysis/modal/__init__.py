"""모달 분석 서브모듈.

공개 API:
    - :class:`DMDAnalyzer`: PyDMD 기반 DMD 분석기
    - :class:`SINDy`: sparse equation discovery wrapper

구현 예정:
    - PyKoopman 래퍼
"""

from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer
from naviertwin.core.flow_analysis.modal.sindy_wrapper import SINDy

__all__ = ["DMDAnalyzer", "SINDy"]
