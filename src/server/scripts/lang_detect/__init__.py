"""다국어 언어감지 평가 하네스 — gold 빌더·후보·혼동행렬·벤치.

gold 평가셋(FLORES-200 dev + 파생 4종)으로 언어 감지 후보(손규칙 /
fastText-LID)를 혼동행렬로 비교한다. 상세는 각 모듈 docstring 참조.
"""

from server.scripts.lang_detect.gold import (
    BUCKET_SPEC, GOLD_PATH, MANIFEST_PATH)

__all__ = ['BUCKET_SPEC', 'GOLD_PATH', 'MANIFEST_PATH']
