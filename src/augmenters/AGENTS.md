# augmenters/

학습 데이터 증강(augmentation) 모듈 모음.

## 서브 모듈

### pii/
합성 PII 주입(injector). 기존 NER 데이터셋(Stockmark, JSONL, HF Hub)의
각 문장에 자연스러운 위치로 합성 PII(전화/주소/생년월일/ID/이메일/카드)를
주입하여 Stockmark 엔티티 + PII 통합 학습 데이터셋을 생성한다.

#### 주요 파일

| 파일 | 역할 |
|------|------|
| `schema.py` | `Entity`, `Record` 공용 dataclass (labelers 호환) |
| `config.py` | `InjectionConfig` (lang, density, pii_labels, label_merge_rules, seed) + `validate()` |
| `injector.py` | `PIIInjector.inject()` / `inject_dataset()` — 접미 삽입 + span 재계산 |
| `label_merger.py` | `NAME→人名`, 단순 지명 `ADDRESS→地名` 병합 규칙 |
| `stats.py` | 라벨별 빈도·커버리지·PII 없는 샘플 비율 리포트 |
| `__main__.py` | `python -m augmenters.pii` CLI 엔트리포인트 |
| `generators/base.py` | `PIIGenerator` Protocol, `get_generator(lang)` factory, 공용 유틸 |
| `generators/ja.py` | 일본어 PII 생성기 (이름/전화/주소/DOB/ID/이메일) |
| `generators/vi.py` | 베트남어 PII 생성기 |
| `loaders/stockmark.py` | Stockmark NER → `Record` 어댑터 |
| `loaders/jsonl.py` | 임의 JSONL(`{text, entities}`) 로더 |
| `loaders/hf.py` | HF Hub 데이터셋 어댑터 (필드 매핑 가능) |

#### 라벨 스키마
- **병합 규칙**: `NAME → 人名`, 단일 지명 `ADDRESS → 地名`
- **신규 PII 라벨**: `PHONE`, `ADDRESS`(복합), `DOB`, `ID_NUMBER`, `EMAIL`, `CREDIT_CARD`
- 최종 라벨: Stockmark 8종 + PII 6종

#### 주입 밀도
기본값 분포 `P(0)=0.2, P(1)=0.4, P(2)=0.3, P(3)=0.1` (문장당 PII 개수).
CLI `--pii-max` 로 상한 조정 가능. 결정론성은 `--seed` 로 보장.

## 사용 예

```bash
# Stockmark 원본에 PII 주입
python -m augmenters.pii --source stockmark --lang ja \
    --output /data/ner/ja_stockmark_pii.jsonl --n-samples 1000

# 임의 JSONL(크롤링 등)에 주입
python -m augmenters.pii --source jsonl --input /data/raw/crawl.jsonl \
    --lang ja --output /data/ner/ja_crawl_pii.jsonl

# HF Hub 데이터셋에 주입
python -m augmenters.pii --source hf --hf-name llm-book/ner-wikipedia-dataset \
    --lang ja --output /data/ner/ja_wiki_pii.jsonl
```

## 통합

생성된 JSONL은 `labelers.ja.JapaneseDatasetLoader.load_local(path)` 로
기존 Stockmark 레코드와 동일 스키마(`{id, text, gold_spans}`)로 로딩된다.

## 테스트
- `tests/augmenters/pii/test_injector.py` — span 일치, 밀도 분포, seed 결정론성, 라벨 병합
- `tests/augmenters/pii/test_label_merger.py` — 병합 규칙 단위 테스트
- `tests/augmenters/pii/test_loader_integration.py` — JSONL → `load_local` 라운드트립
