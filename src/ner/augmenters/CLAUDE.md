# augmenters/

학습 데이터 증강(augmentation) 모듈 모음.

라벨 스키마는 canonical 영문 축약 **10종 평면 목록**
(`docs/manual/data/canonical-entity-schema.md`). NER 5종
(`PER/LOC/ORG/PROD/EVT`) + PII 5종(`DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`).

**LOC/ORG 경계는 언어마다 다르다** — 10종 평면 목록만 공통이다. JA·VI 는 13종 →
10종 평면화 때 인공 시설을 모두 ORG 로 흡수했고, KO 는 narrow-ORG 재정의를 따라
ORG 를 정부·행정·공공·정치 기관으로 좁히고 인공 시설·민간조직을 아예 버린다.
`generators/ko.py` 로 KO 주입을 돌릴 때 이 차이를 JA·VI 기준으로 착각하면
주입 자체는 맞아도 gold 해석이 틀어진다. 언어별 정본: 위 스키마 문서 §KO 및
`docs/manual/data/korean-entity-labeling-rules.md`.

## 서브 모듈

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
| `llm_injector.py` | `LLMInjector` + `VllmClient` — **llm 모드**: LLM이 PII를 자연스럽게 문중에 삽입, 생성 텍스트에서 string match로 span offset 추출; 주입 후 `harden_pii_format_collisions`로 무라벨 CC·ID_NUM 포맷 열을 일관 relabel |
| `stats.py` | 라벨별 빈도·커버리지·PII 없는 샘플 비율 리포트 |
| `verifier.py` | `PIIVerifier` — LLM 교차 검증 (confirmed/missed/conflict 분류, drop_span/drop_record/keep_all 정책). 검증 시 `label_spans(split=False)` 로 호출하여 문맥 보존 |
| `__main__.py` | `python -m ner.augmenters.pii` CLI 엔트리포인트 (`--mode {suffix,llm}`, `--verify vllm` 교차 검증) |
| `loaders.py` | Stockmark / 임의 JSONL / HF Hub → `Record` 어댑터 모음 |
| `generators/base.py` | `generate_pii(label, lang, rng)` 언어·라벨 디스패치 + 공용 유틸(`random_email`·`random_credit_card_number`·`EMAIL_DOMAINS` 등) |
| `generators/ja.py` | 일본어 PII 생성기 (이름/전화/주소/날짜(`generate_dat`)/ID/이메일) |
| `generators/vi.py` | 베트남어 PII 생성기 (이름/전화/주소/날짜(`generate_dat`)/ID/이메일) |
| `generators/ko.py` | 한국어 PII 생성기 (이름/전화(`010`/`02`/지역)/주소/날짜/주민등록번호 — 체크섬 무효로 실유효 번호 비생성) |

#### 라벨 스키마
- **내부 PII 토큰**(생성·병합 전): `NAME`, `PHONE`, `ADDRESS`, `DAT`,
  `ID_NUM`, `EMAIL`, `CREDIT_CARD` (7종 — `ADDRESS`·`NAME`은 병합 대상)
- **병합 규칙** (`DEFAULT_MERGE_RULES`): `NAME → PER`, `ADDRESS → LOC`
  (둘 다 무조건 병합). 학습 데이터에는 `NAME`·`ADDRESS` 라벨이 존재하지
  않는다
- **최종 출력 라벨**: canonical 10종 평면 (NER 5종 `PER/LOC/ORG/PROD/EVT` + PII 5종 `DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`)

#### 주입 밀도
기본값 분포 `P(0)=0.2, P(1)=0.4, P(2)=0.3, P(3)=0.1` (문장당 PII 개수).
CLI `--pii-max` 로 상한 조정 가능. 결정론성은 `--seed` 로 보장.

### wikiann_vi/
WikiANN-vi를 canonical 5종으로 재라벨하는 async LLM 클라이언트 + 품질 측정
유틸 + 코퍼스 빌드 단계. HF 원본(WikiANN 3종 BIO)을 읽는 책임은 본 패키지의
`__main__.py._load_wikiann_hf`에 있으며, 평가용 canonical 덤프
로더(`labelers/vi/dataset_loader.py`)는 HF를 호출하지 않는다.
상세: `docs/manual/pipeline/2-augmentation.md` §2B.

**품질 측정과 코퍼스 빌드를 섞지 말 것** — 앞은 데이터를 안 바꾸고 재라벨이
쓸 만한지만 재고(kappa·Wikidata anchor), 뒤는 학습에 실제로 들어갈 span 을
고르므로 결과가 바뀐다(merge_confidence·silver_gap).

#### 주요 파일

| 파일 | 역할 | 성격 |
|------|------|------|
| `prompts.py` | `SINGLE_PROMPT_TEMPLATE`·`BATCH_PROMPT_TEMPLATE`·`DEFAULT_ENTITY_TYPES`(NER 5종) | — |
| `relabel.py` | `Relabeler`·`parse_spans`·`match_offsets` — async 재라벨 + 응답 파싱·오프셋 정렬 | 재라벨 |
| `__main__.py` | CLI `python -m ner.augmenters.wikiann_vi`. `_load_wikiann_hf` 가 HF 원본 3종 BIO 로딩 담당 | 재라벨 |
| `kappa.py` | Cohen's kappa + agreement 비율 + 타입별 일치·혼동행렬 (span union 기준 pair 수집). CLI: `python -m ner.augmenters.wikiann_vi.kappa` | 품질 측정 |
| `wikidata_anchor.py` | vi.wikipedia 인터링크 → Q-ID → `P31` → canonical 5종 매핑(`WIKIDATA_TO_CANONICAL`), JSON 캐시·요청 간 대기. CLI: `...wikiann_vi.wikidata_anchor` | 품질 측정 |
| `merge_confidence.py` | 정책별 span 선택·필터(`_filter_by_policy`) — `POLICIES` 7종(`recall`·`precision`·`high_only`·`full`·`recall_strict`·`recall_strict_evt`·`recall_strict_prod`). CLI: `...wikiann_vi.merge_confidence --policy <P>` | **코퍼스 빌드** |
| `silver_gap.py` | `apply_silver_gap` — 검증기가 'gold 누락 legit' 으로 독립 확인한 FP→TP 만 골라 gold 에 **additive** 삽입 (leak-free group-kfold 유지) | **코퍼스 빌드** |

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

# 한국어 canonical 10종 gold (PII 4종만, llm 자연삽입, verify 없음)
# KLUE 유래 NER 5종+DAT gold(origin.jsonl)에 PII 4종을 문중 자연삽입.
# verify 미사용: llm extract_spans 가 offset 을 정확 보장(사람 KLUE gold 보존).
python -m ner.augmenters.pii --source jsonl --input data/klue/origin.jsonl \
    --lang ko --pii-labels EMAIL PHONE ID_NUM CREDIT_CARD --mode llm \
    --inject-url http://localhost:8081/v1 \
    --inject-model cyankiwi/gemma-4-31B-it-AWQ-8bit \
    --output data/klue/pii_all.jsonl
```

`--pii-labels` 로 주입 PII 라벨을 제한한다(기본 7종 → 지정 라벨만). 미지정
시 `DEFAULT_PII_LABELS` 전체.

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

생성된 JSONL 의 소비자는 둘이고 목적이 다르다.

- **LLM 벤치마크 평가**: `labelers.ja.JapaneseDatasetLoader.load_local(path)` →
  canonical 라벨 레코드(`{id, text, gold_spans}`)
- **BERT 학습**: `classifier.data_utils.load_jsonl(path)` → 학습용 행. 이쪽은
  라벨이 canonical 10종 안에 있는지 검증하고 위반 시 `ValueError` 로 끊는다.
  증강 산출물이 실제로 학습에 들어가는 지점이라 계약 위반이 여기서 드러난다.

## 테스트
- `tests/ner/augmenters/pii/test_injector.py` — suffix 모드 span 일치, 밀도 분포, seed 결정론성, 라벨 병합
- `tests/ner/augmenters/pii/test_llm_injector.py` — llm 모드 프롬프트 생성, span 추출(string match), 원본 엔티티 재탐색
- `tests/ner/augmenters/pii/test_label_merger.py` — 병합 규칙 단위 테스트 (`injector.apply_label_merge` 대상)
- `tests/ner/augmenters/pii/test_loader_integration.py` — JSONL → `load_local` 라운드트립
- `tests/ner/augmenters/pii/test_verifier.py` — 교차 검증 (confirmed/missed/conflict, 정책별 동작, 부분 매칭, 데이터셋 리포트)
- `tests/ner/augmenters/pii/test_ko_injection.py` · `test_vi_injection.py` — 언어별 PII 생성기·주입기 동작
- `tests/ner/augmenters/pii/test_main_verify_dispatch.py` — `__main__._build_verify_labeler` 의 lang 분기
- `tests/ner/augmenters/wikiann_vi/` — 재라벨 파서·offset 매칭·kappa·Wikidata anchor·confidence 병합·silver-갭 additive 삽입(`test_audit_silver_gap.py`)
