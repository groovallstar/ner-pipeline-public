# Ramshaw & Marcus (1995) — Text Chunking using Transformation-Based Learning

- **저자**: Lance A. Ramshaw, Mitchell P. Marcus
- **연도**: 1995
- **학회**: Third Workshop on Very Large Corpora (VLC), ACL
- **링크**: https://aclanthology.org/W95-0107/
- **유형**: 1차 문헌 (방법론 원조)

## 핵심 요지

텍스트 청킹(명사구·동사구 등 단순 구조 인식) 작업에 **변환 기반 학습(Transformation-Based Learning)**을 적용한 연구. 본 위키의 관점에서 더 중요한 기여는 청킹 경계를 표현하기 위한 **IOB(=BIO) 태깅 스킴**을 제안·정착시킨 것이다.

## BIO 스킴 정의 (이 논문에서 도입)

각 토큰에 다음 셋 중 하나를 부여:
- `B-<TYPE>` — 청크의 시작
- `I-<TYPE>` — 같은 청크의 내부 (직전 토큰과 동일 청크에 속함)
- `O` — 청크 외부

이 단순한 표기법으로 시퀀스 라벨링 모델이 임의 길이의 다중 청크를 처리할 수 있게 되었고, 이후 **NER, chunking, 슈퍼태깅 등 시퀀스 인식 전반의 사실상 표준**이 되었다.

## 인용하는 위키 페이지

- `concepts/bio-tagging.md` — BIO ↔ Span 변환 알고리즘의 전제
