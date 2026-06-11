# augmenters/

학습 데이터 증강(augmentation) 모듈 모음.

라벨 스키마는 canonical 영문 축약 **10종 평면 목록**
(`docs/manual/data/canonical-entity-schema.md`). NER 5종
(`PER/LOC/ORG/PROD/EVT`) + PII 5종(`DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`).
JA·VI 공통 스키마이며, 13종 → 10종 평면화 후 LOC/ORG 경계가 재정의돼
인공 시설은 모두 ORG로 분류된다.

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
WikiANN-vi를 canonical 5종으로 재라벨하는 async LLM 클라이언트
+ 검증 유틸(kappa·Wikidata anchor·confidence 병합). HF 원본(WikiANN 3종
BIO)을 읽는 책임은 본 패키지의 `__main__.py._load_wikiann_hf`에 있으며,
평가용 canonical 덤프 로더(`labelers/vi/dataset_loader.py`)는 HF를
호출하지 않는다. 상세: `docs/manual/data/vietnamese-ner.md`.

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
