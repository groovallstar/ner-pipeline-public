# Park et al. (2021) — KLUE: Korean Language Understanding Evaluation

- **저자**: Sungjoon Park, Jihyung Moon, Sungdong Kim, Won Ik Cho 외 다수 (총 31명)
- **연도**: 2021 (제출 5월, 최종 11월)
- **출판**: arXiv:2105.09680 (cs.CL); NeurIPS 2021 Datasets & Benchmarks
- **링크**: https://arxiv.org/abs/2105.09680
- **벤치마크 사이트**: https://klue-benchmark.com
- **유형**: 1차 문헌 (데이터셋·벤치마크 명세)

## 핵심 요지

한국어 NLU 종합 벤치마크(KLUE)를 제안. 8개 태스크 — Topic Classification, Semantic Textual Similarity, NLI, **Named Entity Recognition (KLUE-NER)**, Relation Extraction, Dependency Parsing, MRC, Dialogue State Tracking — 와 함께 KLUE-BERT / KLUE-RoBERTa 사전학습 모델을 공개.

## NER 부분 (본 프로젝트 관점)

- **태그 체계**: 6종 — `PS`(Person), `LC`(Location), `OG`(Organization), `DT`(Date), `TI`(Time), `QT`(Quantity)
- **토큰화 단위**: **음절 단위(syllable-level)** + 공백 토큰 — 이는 본 위키의 `concepts/bio-tagging.md` §2(음절/어절/형태소)와 §4.2 처리 단계의 직접 근거.
- **데이터 출처**: 위키트리 뉴스 + 에어비앤비 리뷰 등.
- 음절 단위 태깅이라 조사 분리가 가능하다는 특성이 본 파이프라인의 span 변환 로직(`extract_spans_from_bio`, `joiner=""`) 설계에 반영되어 있음.

## 인용하는 위키 페이지

- `concepts/bio-tagging.md` — 음절 단위 사례
