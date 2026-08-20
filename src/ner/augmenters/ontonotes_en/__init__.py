"""OntoNotes5(영문) → canonical 10 종 평면 변환.

원본은 사전 토큰화된 18 타입 BIO 이고, 이 패키지는 그것을 자연문 +
char-offset span 의 canonical 라벨 JSONL 로 옮긴다. PII 4 종
(EMAIL/PHONE/ID_NUM/CREDIT_CARD)은 원본에 없어 `augmenters.pii` 가
이어서 주입한다 — `DAT` 은 원본 `DATE` 가 gold 로 주므로 주입 대상이 아니다
(KO 와 같은 패턴).

| 모듈 | 역할 |
|---|---|
| `mapping` | 원본 18 타입 → canonical 매핑표 + 전수성 게이트 |
| `detokenize` | 토큰 배열 → 자연문 복원 + 토큰별 char-offset |
| `convert` | BIO 디코드 · 레코드 변환 · 엔티티↔토큰 대조 |
| `__main__` | CLI (`python -m ner.augmenters.ontonotes_en`) |
"""
