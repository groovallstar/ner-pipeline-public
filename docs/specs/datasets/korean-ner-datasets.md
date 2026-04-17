# 한국어 NER 대체 데이터셋 조사

> 조사 일자: 2026-04-09
> 목적: KLUE NER 외 한국어 NER 벤치마크/파인튜닝용 데이터셋의 특성과 현재 파이프라인 호환성 분석
> 관련 코드: `src/labelers/dataset_loader.py`, `src/labelers/tag_aligner.py`

## 목차

1. [조사 배경](#1-조사-배경)
2. [주요 한국어 NER 데이터셋 목록](#2-주요-한국어-ner-데이터셋-목록)
3. [토큰화 단위 및 BIO 태그 형식 검증](#3-토큰화-단위-및-bio-태그-형식-검증)
4. [WikiANN 도입 시 주의점](#4-wikiann-도입-시-주의점)

---

## 1. 조사 배경

현재 파이프라인은 KLUE NER 1개 데이터셋에만 의존한다. 한국어 NER 모델의 일반화 성능을 측정하려면 독립적인 추가 벤치마크 데이터셋이 필요하다. 이 문서는 다음 질문에 답한다.

- KLUE 외 어떤 한국어 NER 데이터셋이 공개되어 있는가?
- 각 데이터셋의 토큰화 단위(음절/어절/형태소)와 태그 형식은 무엇인가?
- 현재 음절 단위 BIO 파이프라인과 직접 호환되는가?

> **참고:** 토큰화 단위 차이가 LLM 라벨링 검증에 미치는 영향, BIO ↔ Span 변환 알고리즘, 단위별 span 추출 과정은 별도 문서 `docs/wiki/concepts/bio-tagging.md`에서 다룬다.

---

## 2. 주요 한국어 NER 데이터셋 목록

### HuggingFace에서 바로 로드 가능

| 데이터셋 | HF 경로 | 엔티티 | 규모 | 라이선스 |
|----------|---------|--------|------|----------|
| **WikiANN (ko)** | `tner/wikiann` (config: `ko`) | 3종 (PER/LOC/ORG) | 40K 문장 (train 20K, val 10K, test 10K) | CC BY-SA 3.0 |
| **KMOU NER** | `nlp-kmu/kor_ner` | 5종 (PS/LC/OG/DT/TI) | ~수천 | MIT (비상업) |
| **MultiCoNER v2** | `MultiCoNER/multiconer_v2` | 33종 (6대 범주) | 수십만 (전체 언어) | CC BY 4.0 |

> **주의:** MultiCoNER v2는 HuggingFace Hub 기준 한국어(KO) config를 지원하지 않는다. 지원 언어: `BN, ZH, EN, FA, FR, DE, HI, IT, MULTI, PT, ES, SV, UK`.

### 신청/직접 다운로드 필요

| 데이터셋 | 접근 방법 | 엔티티 | 규모 | 특징 |
|----------|-----------|--------|------|------|
| **Naver x 창원대 NER** | Korpora 패키지 / `connectfoundation/naverconnect-dataset-ner` | 6종 (KLUE와 동일: PS/LC/OG/DT/TI/QT) | 21,008 문장 | KLUE 이전 가장 많이 사용된 데이터셋. KLUE NER의 원천 소스 |
| **모두의 말뭉치 NE** | corpus.korean.go.kr 신청 | 15대분류 / 150소분류 (TTA 표준) | 600만 어절 | 국내 연구 표준. 가장 세밀한 분류 |
| **KONNE/KONEC** | GitHub `korean-named-entity/konne` | 150종 (중첩 NER) | 26,008 문장 | KLUE 기반 중첩 개체명 확장 (2022) |
| **AI Hub NER** | aihub.or.kr 신청 | 4/15/150종 | 대규모 | 국내 상용 모델 학습 주력 |
| **KBMC** | 논문 공개 예정 (arXiv:2403.16158) | 3종 (Disease/Body/Treatment) | 미공개 | 한국어 최초 오픈소스 의료 NER (LREC-COLING 2024) |

---

## 3. 토큰화 단위 및 BIO 태그 형식 검증

실제 HuggingFace에서 로드하여 샘플을 직접 확인한 결과.

### 3.1 WikiANN (ko)

- **토큰 단위: 어절** (공백으로 분리된 단어)
- 평균 토큰 길이: 2.76자, 1글자 토큰 27.9%, 3글자 이상 49.4%
- **태그 형식: IOB2 (표준 BIO)**
- 태그 레이블: `B-LOC`, `B-ORG`, `B-PER`, `I-LOC`, `I-ORG`, `I-PER`, `O`

**실제 예시:**
```
'현재'        -> O
'대한민국'    -> B-LOC        # 어절 통째로 태깅
'K리그'       -> B-ORG
'챌린지의'    -> I-ORG        # 조사 '의'도 어절 경계 내에 포함
'서울'        -> B-ORG
'이랜드'      -> I-ORG
'FC에서'      -> I-ORG
```

### 3.2 KMOU NER (`nlp-kmu/kor_ner`)

- **토큰 단위: 형태소**
- 토큰 예시에 단독 자음(`'ㄴ'`, `'ㄹ'`), 조사(`'으로'`, `'에서'`) 포함
- **태그 형식: 변형 BIO (비표준)** — `I` 태그가 엔티티 타입을 구분하지 않음

**실제 예시:**
```
'이토'      -> B_PS
'히로부미'  -> I            # 엔티티 타입 없이 'I' 단독
'안중근'    -> B_PS
'만주'      -> B_LC
'할빈'      -> B_LC
```

### 3.3 KLUE NER (현재 사용 중, 참고)

- **토큰 단위: 음절** (100% 1글자, 공백도 별도 토큰)
- **태그 형식: IOB2 (표준 BIO)**
- 한국어 NER 데이터셋 중 **유일하게 음절 단위**

**실제 예시:**
```
'영' -> B-LC
'동' -> I-LC
'고' -> I-LC
'속' -> I-LC
'도' -> I-LC
'로' -> I-LC
' '  -> O        # 공백도 O 태그
'강' -> B-LC
'릉' -> I-LC
```

### 3.4 요약 비교표

| 데이터셋 | 토큰 단위 | 태그 형식 | 현재 파이프라인 직접 호환 |
|----------|----------|-----------|------------------------|
| WikiANN (ko) | 어절 (평균 2.76자) | IOB2 표준 | 변환 필요 (어절→음절) |
| MultiCoNER v2 | — | — | 한국어 미지원 |
| KMOU NER | 형태소 | 변형 BIO (비표준) | 변환 필요 (형태소→음절 + 태그 재매핑) |
| KLUE NER | 음절 | IOB2 표준 | **직접 호환** |

**핵심 발견:** 조사한 한국어 NER 데이터셋 중 **음절 단위 BIO를 사용하는 것은 KLUE NER이 유일**하다.

---

## 4. WikiANN 도입 시 주의점

- **엔티티 커버리지 축소**: WikiANN은 3종(PER/LOC/ORG)만 있어 KLUE 6종 중 DT/TI/QT가 빠진다. 같은 프롬프트로 평가하면 DT/TI/QT 예측은 모두 FP(false positive)로 처리되므로 다음 중 하나가 필요하다:
  - 평가 시 LLM 출력에서 3종 외 엔티티를 필터링
  - WikiANN 전용 프롬프트를 3종으로 제한
- **어절 경계 문제**: WikiANN은 `'대한민국'` 같은 어절 전체가 한 토큰이지만 `'챌린지의'`처럼 조사가 어절 경계 내에 포함되는 경우가 있다. LLM이 `'챌린지'`만 ORG로 잡으면 gold(`'챌린지의'`)와 strict match에서 불일치한다.

---

## 참고 문헌

- [tner/wikiann on HuggingFace](https://huggingface.co/datasets/tner/wikiann)
- [nlp-kmu/kor_ner on HuggingFace](https://huggingface.co/datasets/nlp-kmu/kor_ner)
- [MultiCoNER/multiconer_v2 on HuggingFace](https://huggingface.co/datasets/MultiCoNER/multiconer_v2)
- [NAVER x Changwon NER — Korpora](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/naver_changwon_ner.html)
- [모두의 말뭉치 개체명 — Korpora](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/modu_ne.html)
- [KONNE GitHub](https://github.com/korean-named-entity/konne)
- [KBMC paper — arXiv:2403.16158](https://arxiv.org/abs/2403.16158)
