"""공개 도메인 데이터 마이닝 + canonical 재라벨 파이프라인 (JA PROD 보강).

PROD 가 약한 4도메인 (법령·서적·교통카드·악곡) 의 공개 데이터에서 실제
문장을 마이닝해, 다중 LLM 교차검증으로 canonical 10종 평면 재라벨한 뒤
classifier contract JSONL 로 출력한다 (train extra + test 확장, leak-free).

서브모듈:
- schema: 공용 스키마·변환·검증 (순수 함수, 외부 의존 없음)
"""
