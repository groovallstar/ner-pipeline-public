# augmenters/

학습 데이터 증강(augmentation) 모듈 모음.

라벨 스키마는 canonical 영문 축약 **10종 평면 목록**
(`docs/manual/data/canonical-entity-schema.md`). NER 5종
(`PER/LOC/ORG/PROD/EVT`) + PII 5종(`DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`).
JA·VI 공통 스키마이며, 13종 → 10종 평면화 후 LOC/ORG 경계가 재정의돼
인공 시설은 모두 ORG로 분류된다.

## 서브 모듈

### crawlers/
뉴스 RSS 크롤러 + NER BIO 태깅 파이프라인. Phase-1 스모크 테스트 버전이며
Yonhap(연합뉴스) 한국어 소스를 지원한다. 상세: `crawlers/ko/AGENTS.md`.

```bash
python -m ner.augmenters.crawlers.ko \
    --source yna --max-sentences 100 \
    --output-dir data/ner/raw \
    --vllm-base-url http://localhost:8081/v1 \
    --model Qwen/Qwen3.5-27B
```

Flat `CrawlerSpec` dataclass 스타일(상속 없음). 출력은 `data/ner/raw/{source}/`
(`.gitignore` 대상). 기존 `labelers.ko.VllmNERLabeler.label_spans`를 문장 단위로
호출하며 `BaseVllmLabeler`는 수정하지 않는다.

### pii/
합성 PII 주입(injector). 기존 NER 데이터셋(Stockmark, JSONL, HF Hub)의
각 문장에 자연스러운 위치로 합성 PII(전화/주소/생년월일/ID/이메일/카드)를
주입하여 NER + PII 통합 학습 데이터셋을 생성한다.

#### 주요 파일

| 파일 | 역할 |
|------|------|
| `schema.py` | `Entity`, `Record` 공용 dataclass (labelers 호환) |
| `config.py` | `InjectionConfig` (lang, density, pii_labels, label_merge_rules, seed) + `validate()` |
| `injector.py` | `PIIInjector` — **suffix 모드**: 문장 끝 접미 삽입 + span 재계산. `apply_label_merge`/`merge_entities` 규칙 기반 병합(`NAME→PER`, `ADDRESS→LOC` 무조건)도 제공 |
| `llm_injector.py` | `LLMInjector` + `VllmClient` — **llm 모드**: LLM이 PII를 자연스럽게 문중에 삽입, 생성 텍스트에서 string match로 span offset 추출 |
| `stats.py` | 라벨별 빈도·커버리지·PII 없는 샘플 비율 리포트 |
| `verifier.py` | `PIIVerifier` — LLM 교차 검증 (confirmed/missed/conflict 분류, drop_span/drop_record/keep_all 정책). 검증 시 `label_spans(split=False)` 로 호출하여 문맥 보존 |
| `__main__.py` | `python -m ner.augmenters.pii` CLI 엔트리포인트 (`--mode {suffix,llm}`, `--verify vllm` 교차 검증) |
| `loaders.py` | Stockmark / 임의 JSONL / HF Hub → `Record` 어댑터 모음 |
| `generators/base.py` | `PIIGenerator` Protocol, `get_generator(lang)` factory, 공용 유틸 |
| `generators/ja.py` | 일본어 PII 생성기 (이름/전화/주소/날짜(`generate_dat`)/ID/이메일) |
| `generators/vi.py` | 베트남어 PII 생성기 (이름/전화/주소/날짜(`generate_dat`)/ID/이메일) |

#### 라벨 스키마
- **내부 PII 토큰**(생성·병합 전): `NAME`, `PHONE`, `ADDRESS`, `DAT`,
  `ID_NUM`, `EMAIL`, `CREDIT_CARD` (7종 — `ADDRESS`·`NAME`은 병합 대상)
- **병합 규칙** (`DEFAULT_MERGE_RULES`): `NAME → PER`, `ADDRESS → LOC`
  (둘 다 무조건 병합). 학습 데이터에는 `NAME`·`ADDRESS` 라벨이 존재하지
  않는다
- **최종 출력 라벨**: canonical 13종 (NER 8종 + `DAT` 1종 + PII 4종)

#### 주입 밀도
기본값 분포 `P(0)=0.2, P(1)=0.4, P(2)=0.3, P(3)=0.1` (문장당 PII 개수).
CLI `--pii-max` 로 상한 조정 가능. 결정론성은 `--seed` 로 보장.

### wikiann_vi/
WikiANN-vi를 canonical 5종으로 재라벨하는 async LLM 클라이언트
+ 검증 유틸(kappa·Wikidata anchor·confidence 병합). HF 원본(WikiANN 3종
BIO)을 읽는 책임은 본 패키지의 `__main__.py._load_wikiann_hf`에 있으며,
평가용 canonical 덤프 로더(`labelers/vi/dataset_loader.py`)는 HF를
호출하지 않는다. 상세: `docs/manual/data/vietnamese-ner.md`.

### ja/ — JA classifier 천장 회복 도구 모음

JA NER classifier 의 잔여 미달 클래스 (PROD recall · ORG/PROD precision)
회복을 위한 oversample 도구 서브패키지 모음. classifier
`error_analysis.py --with-diagnosis` 산출물을 입력으로 받아 leak-free
보강 JSONL 을 생성한다. 학습 시 `classifier --data-extra-train-jsonl`
옵션과 조합해 train split 에만 합치고 valid/test 는 원본 split 그대로
유지하는 평가 무결성을 지킨다.

#### ja/negative/ — 환각 부정 예시 oversample

환각 (FP HALLUCINATION) 감소용. 환각 surface 를 학습 셋에서 entity
라벨로 표시되지 않은 위치에 가진 문장을 N 배 oversample 한다.

| 파일 | 역할 |
|------|------|
| `oversampler.py` | `extract_seeds` / `filter_ambiguous_seeds` / `find_candidate_indices` / `oversample_to_jsonl` 핵심 로직 |
| `__main__.py` | `python -m ner.augmenters.ja.negative` CLI (`--auto-exclude-ambiguous`, `--seed-pred-types`, `--extra-only`) |

사용 예 (production 셋업, leak-free):

```bash
# 1. 보강 jsonl 생성 (extra-only: classifier --data-extra-train-jsonl 용)
python -m ner.augmenters.ja.negative \
    --input data/stockmark/pii_all.jsonl \
    --diagnosis results/classifier/ja_sweep/v3_boundary/error_analysis.json \
    --output data/stockmark/pii_neg_aug_N2_extra.jsonl \
    --oversample 2 --auto-exclude-ambiguous --extra-only

# 2. classifier 학습 (leak-free: train 에만 합쳐짐)
python -m ner.classifier --lang ja \
    --data data/stockmark/pii_all.jsonl \
    --data-extra-train-jsonl data/stockmark/pii_neg_aug_N2_extra.jsonl \
    --boundary-b-weight 1.5 --boundary-i-weight 1.2
```

핵심 옵션:
- `--oversample N` (default 2): 각 후보 문장 등장 횟수. extra-only 시
  (N-1) 번 출력
- `--auto-exclude-ambiguous`: 학습 셋에 entity 로 등장한 surface 자동
  제외 (recall 회귀 차단)
- `--seed-pred-types PER,LOC,ORG,PROD,EVT`: 환각 pred type 필터
- `--extra-only`: extra 만 출력 (leak-free 평가 필수)

#### ja/prod_seed/ — PROD 도메인 휴리스틱 seed oversample

PROD recall 의 long-tail 도메인 (법안·서적·식품·교통·음악) 신호 보강용.
random PROD-positive oversample 은 surface 분포가 famous media·software 에
편중되어 test PROD FN 과 substring overlap 0% → 천장 그대로. 본 도구는
도메인 suffix·키워드 정규식 (`DOMAIN_PATTERNS`) 으로 train+valid PROD-pos
문장을 정밀 선별·N 배 oversample 한다.

| 파일 | 역할 |
|------|------|
| `seed_selector.py` | `DOMAIN_PATTERNS` (law/book/food/transit_card/music_work 5종) / `categorize_prod_surface` / `select_domain_seed_indices` / `select_long_seed_indices` / `oversample_to_jsonl` |
| `__main__.py` | `python -m ner.augmenters.ja.prod_seed` CLI (`--include-domain`, `--include-long`, `--base-extra`, `--oversample`, `--extra-only`) |

사용 예 (PROD 도메인 + base extra = S7N2 negative):

```bash
python -m ner.augmenters.ja.prod_seed \
    --input data/stockmark/pii_all_phonediv.jsonl \
    --base-extra data/stockmark/pii_neg_aug_N2_extra.jsonl \
    --output data/stockmark/pii_extra_s8_prod_domain_N5.jsonl \
    --oversample 5 --extra-only
```

핵심 옵션:
- `--include-domain` (default on): law/book/food/transit_card/music_work
  5종 도메인 휴리스틱 매칭 seed 풀
- `--include-long` (default off): 도메인 미매칭이지만 LONG (`--long-min-length`
  default 6) surface 풀 (`DEFAULT_LONG_EXCLUDE_KEYWORDS` 로 famous media 제외)
- `--base-extra PATH`: prefix 로 prepend 할 기존 extra (예: ja.negative 의
  출력) — 두 보강을 한 jsonl 로 합쳐 classifier 에 단일 주입
- `--oversample N` (default 5): seed-major 패턴으로 각 후보 N 회 연속 출력
  (HF Trainer shuffle 입력 순서 결정성 유지)
- `--extra-only` (default on): leak-free 평가 필수

#### domain_mine/ — 공개데이터 마이닝 + 교차검증 재라벨 (post-#61, 이슈 #64)

prod_seed 가 *기존 데이터* 에서 도메인 PROD 를 선별하는 반면, domain_mine
은 *공개 데이터* (e-Gov 법령·Wikidata 작품·악곡·교통카드) 명칭을 모아
ja.wikipedia 에서 문장을 마이닝하고 Gemma+Qwen 2모델 합의로 canonical 10종
재라벨해 신규 도메인 PROD 데이터를 생성한다.

| 파일 | 역할 |
|------|------|
| `schema.py` | contract 변환(span→entity)·offset 무결성 검증·도메인 상수 (순수) |
| `name_sources.py` | e-Gov(XML)·Wikidata SPARQL·curated 명칭 수집 + 캐시 CLI |
| `sentence_miner.py` | ja.wiki 전문검색→문장분리→명칭 포함 문장+offset 추출 |
| `relabel.py` | 2모델 재라벨 + merge_confidence 합의 + `apply_anchor` (confirmed/conflict/anchor_only) |
| `build_extra.py` | status 분배(confirmed→test, anchor_only→train) + `--confirmed-test-frac` 분할 + leak-free dedup |

> ⚠️ **이슈 #64 결론**: 본 도메인 데이터를 train 에 증강하면 *도메인 편중
> test* 에선 PROD P 가 오르지만 *운영 원본 test* 에선 회귀(과적합, #45 재현).
> production 모델 학습 기본 레시피엔 미포함. 상세:
> `docs/reports/ja-prod-gate-public-data-relabel-2026-05.md`.

## 사용 예

```bash
# suffix 모드 (규칙 기반, 결정론적, LLM 불필요)
python -m ner.augmenters.pii --source stockmark --lang ja \
    --output data/stockmark/pii_test.jsonl --n-samples 1000

# llm 모드 (자연 삽입, vLLM 필요) + 교차 검증
python -m ner.augmenters.pii --source stockmark --lang ja \
    --output data/stockmark/pii_test.jsonl --n-samples 1000 \
    --mode llm \
    --vllm-url http://localhost:8081/v1 \
    --vllm-model Qwen/Qwen3.5-27B \
    --verify vllm --verify-policy drop_span

# 임의 JSONL(크롤링 등)에 주입
python -m ner.augmenters.pii --source jsonl --input /data/raw/crawl.jsonl \
    --lang ja --output /data/ner/ja_crawl_pii.jsonl

# HF Hub 데이터셋에 주입
python -m ner.augmenters.pii --source hf --hf-name llm-book/ner-wikipedia-dataset \
    --lang ja --output /data/ner/ja_wiki_pii.jsonl
```

## 모드 선택 가이드

| 항목 | suffix | llm |
|------|--------|-----|
| 삽입 위치 | 문장 끝 접미 | 문장 중간 자연 |
| 결정론성 | O (seed 기반) | X (LLM 생성) |
| LLM 필요 | X | O (vLLM) |
| span 정확성 | 코드 계산 보장 | string match + 원본 drop 가능 |
| 교차 검증 가치 | 낮음 (이미 보장) | 높음 (문맥 깨질 수 있음) |
| BERT 학습 적합성 | 낮음 (패턴 편향) | 높음 (다양한 문맥) |

## 통합

생성된 JSONL은 `labelers.ja.JapaneseDatasetLoader.load_local(path)` 로
canonical 라벨 레코드(`{id, text, gold_spans}`)로 로딩된다.

## 테스트
- `tests/ner/augmenters/pii/test_injector.py` — suffix 모드 span 일치, 밀도 분포, seed 결정론성, 라벨 병합
- `tests/ner/augmenters/pii/test_llm_injector.py` — llm 모드 프롬프트 생성, span 추출(string match), 원본 엔티티 재탐색
- `tests/ner/augmenters/pii/test_label_merger.py` — 병합 규칙 단위 테스트 (`injector.apply_label_merge` 대상)
- `tests/ner/augmenters/pii/test_loader_integration.py` — JSONL → `load_local` 라운드트립
- `tests/ner/augmenters/pii/test_verifier.py` — 교차 검증 (confirmed/missed/conflict, 정책별 동작, 부분 매칭, 데이터셋 리포트)
- `tests/ner/augmenters/wikiann_vi/` — 재라벨 파서·offset 매칭·kappa·Wikidata anchor·confidence 병합
