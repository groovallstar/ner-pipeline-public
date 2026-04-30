# Wikidata Anchor Verification

LLM이 붙인 NER 라벨이 외부 지식 베이스(Wikipedia·Wikidata) 기준과 얼마나 정합한지 **사후(post-hoc) 검증**하는 모듈. WikiANN-vi 재라벨 품질을 독립 소스로 확인하려고 도입됐다.

- 소스: `src/ner/augmenters/wikiann_vi/wikidata_anchor.py`
- 실행: `python -m ner.augmenters.wikiann_vi.wikidata_anchor`
- 관련 리포트: `docs/reports/vietnamese-ner-schema-expansion-2026-04.md` §5

## 배경

LLM 재라벨(`Relabeler`)은 LLM 파라미터 지식만으로 타입을 결정한다. 이 결과를 그대로 신뢰하려면 LLM 자신과 무관한 두 번째 증거가 필요하다. Wikipedia는 사람이 유지하는 백과사전이고, Wikidata의 P31(`instance of`)은 해당 엔티티의 "종류"를 구조화해 저장한 속성이다. 둘을 엮으면 LLM 판단의 독립 "닻(anchor)"이 된다.

중요 원칙:
- **검증은 라벨을 바꾸지 않는다.** 통계(일치율·per-type agreement)와 불일치 샘플을 리포트로 남길 뿐, 원본 JSONL은 불변이다.
- **매핑된 엔티티만 집계된다.** Wikipedia에 페이지가 없거나 P31이 비어 있거나 `WIKIDATA_TO_CANONICAL` 테이블에 없는 Q-ID는 검증에서 제외된다 (10K에서 약 49%만 검증 가능).
- **앵커 라벨 공간은 canonical 5종**(PER/LOC/ORG/PROD/EVT). 이슈 #21에서 8종(PER/LOC/FAC/CORP/PROD/EVT/POL/ORG) → 5종으로 축소되었고, 이슈 #27에서 시설 카테고리(`FAC`)가 LOC가 아닌 ORG로 재배치됐다. 테이블 내 섹션 헤더(`# FAC`, `# CORP`, `# POL`)는 이력 추적용으로 남아 있되 매핑 값은 모두 5종 중 하나다.

## 조사 범위 — 무엇을 조사하고 무엇을 조사하지 않는가

이 모듈은 **베트남어 Wikipedia/Wikidata 전체를 크롤·스캔하지 않는다.** 조사 대상은 재라벨 JSONL 안에 실제로 등장한 span뿐이며, 외부 호출은 다음 두 가지로만 제한된다:

1. **입력 데이터셋에 등장한 unique 표면형 목록** → Wikipedia `pageprops` 조회로 해당 문자열에 매칭되는 Q-ID만 수집
2. **그렇게 얻은 Q-ID 집합** → Wikidata `wbgetentities` 조회로 각 엔티티의 P31만 수집

즉 "베트남어 엔티티 전수 조사"가 아니라 **"우리 데이터셋에 실제로 쓰인 문자열이 Wikipedia 기준으로 어떤 타입으로 간주되는가"** 를 묻는 구조다. 데이터셋에 없는 지명·인물은 쿼리 자체가 발생하지 않는다 (10K 기준 unique 7,647개만 조회).

또한 **수작업 매핑 테이블(`WIKIDATA_TO_CANONICAL`, 현재 약 163개 Q-ID)은 엔티티 Q-ID가 아니라 "타입" Q-ID(= P31 값) 목록이다.** 구분이 중요하다:

| 구분 | 예 | 테이블에 넣는가 |
|---|---|---|
| 엔티티 Q-ID | `Q1858` (Hà Nội), `Q8447` (Hồ Chí Minh) | ❌ 넣지 않음 |
| 타입 Q-ID (P31 값) | `Q515` (city), `Q5` (human), `Q43229` (organization) | ✅ 5종으로 매핑 |

따라서 데이터셋에 새로운 지명(예: 특정 마을)이 등장해도 그 엔티티를 테이블에 일일이 등재할 필요가 없다. 해당 엔티티의 P31 값(예: `Q515` city)이 이미 테이블에 있으면 자동으로 `LOC`로 앵커링된다. 매핑 테이블 확장은 **새로운 *타입*이 데이터에 등장했을 때만** 필요하다(예: `Q190752` 같은 unmapped 타입 Q-ID가 누적 빈도 상위에 오면 추가 검토).

요컨대 — 스키마(= 5종 × 대응 타입 Q-ID)만 수작업으로 관리하고, 개별 엔티티 판정은 Wikipedia/Wikidata의 P31을 그대로 위탁하는 구조다.

## 전체 파이프라인

```
재라벨 JSONL (gold_spans_8type)
  └─ 표면형 수집 (unique)
       └─ [1] vi.wikipedia.org/w/api.php   (pageprops)
            └─ Q-ID
                 └─ [2] www.wikidata.org/w/api.php (wbgetentities)
                      └─ P31 Q-ID 리스트
                           └─ [3] WIKIDATA_TO_CANONICAL 테이블
                                └─ canonical 5종 추론 타입
                                     └─ [4] LLM 라벨과 비교
                                          └─ 일치율·per-type agreement 집계
```

## 단계별 상세

### 1단계: 표면형 → Wikipedia → Q-ID

`fetch_qids(titles)` (`wikidata_anchor.py`)

- 엔드포인트: `https://vi.wikipedia.org/w/api.php`
- 파라미터: `action=query&prop=pageprops&ppprop=wikibase_item&redirects=1`
- 배치 단위: 최대 50 제목 / 요청 (`_BATCH = 50`)
- 호출 간 `0.2s` sleep (Wikimedia 레이트 리밋 관례)
- 동일 페이지로 향하는 리다이렉트·정규화 체인은 `_resolve_title()` 로 역추적해 원 표면형에 Q-ID를 매핑

반환: `{표면형: Q-ID | None}`. 없는 페이지는 `None`.

예: `'Hà Nội' → 'Q1858'`, `'Ốc móng tay' → None` (페이지 없음).

### 2단계: Q-ID → P31(instance of)

`fetch_p31(qids)` (`wikidata_anchor.py`)

- 엔드포인트: `https://www.wikidata.org/w/api.php`
- 파라미터: `action=wbgetentities&props=claims`
- 배치 50 Q-ID / 요청, 동일하게 `0.2s` sleep
- 응답 JSON에서 `entities[qid].claims.P31[*].mainsnak.datavalue.value.id` 경로로 P31 Q-ID 리스트 추출

반환: `{Q-ID: [P31 Q-ID, ...]}`. 한 엔티티가 여러 P31을 가질 수 있다(예: 수도이자 도시).

예: `'Q1858' → ['Q5119', 'Q515']` (capital city, city).

### 3단계: P31 → 5종 매핑

`anchor_type(p31_qids)` (`wikidata_anchor.py`)

- `WIKIDATA_TO_CANONICAL` 수작업 큐레이션 테이블에서 조회 (현재 약 163개 Q-ID)
- P31 리스트를 **순차 스캔해 첫 매칭**을 반환. 여러 P31 중 하나라도 5종에 매핑되면 그 타입 채택
- 테이블 섹션 헤더는 축소 전 8종 분류(`# PER`, `# LOC`, `# FAC`, `# CORP`, `# PROD`, `# EVT`, `# POL`, `# ORG`)를 이력 추적용으로 보존하되, 매핑 값은 모두 5종(`PER/LOC/ORG/PROD/EVT`) 중 하나로 축소돼 있다(`FAC/CORP/POL → ORG`).

예: `['Q5119', 'Q515'] → 'LOC'` (Q515=city가 테이블에 있음).

**테이블 확장 정책**: 상위 빈도 unmapped Q-ID를 리포트에 남겨 후속 이슈에서 수작업 추가. 메타/비엔티티(`Q4167410` disambiguation, `Q16521` taxon 등)는 의도적으로 미매핑.

### 4단계: LLM 라벨 vs 앵커 타입 비교

`run_anchor(records, span_key, cache_path)` (`wikidata_anchor.py`)

각 span `(surface, pred_type, record_id)` 에 대해:

1. `qid_cache[surface]` 조회 → 없으면 제외
2. `p31_cache[qid]` 조회 → 빈 리스트면 제외
3. `anchor_type(p31)` 호출 → None이면 미매핑으로 집계 후 제외
4. `anchor == pred_type` 이면 match, 아니면 mismatch

산출 지표:
- `matches`, `mismatches`, `agreement = matches / with_mapped_type`
- `per_type_agreement[type] = {agreed, total, ratio}`
- `unmapped_qids` (상위 20 — 테이블 확장 판단용)
- `mismatch_samples` (상위 30 — 수작업 분석용)

## 캐시 설계

- 캐시 파일: `data/wikiann_vi/wikidata_cache.json` (기본값, `--cache`로 변경 가능)
- 스키마: `{'qid': {표면형: Q-ID}, 'p31': {Q-ID: [P31, ...]}}`
- 실행 시 캐시에 없는 항목만 새로 네트워크 호출. 매핑 테이블(3단계)은 코드에 내장되어 있어 테이블 확장 후 재실행해도 네트워크 재호출 없이 재집계 가능 → 10K 앵커 재측정(v1→v2)이 분 단위로 가능

리포트 §5.5의 "캐시 재활용으로 매핑 테이블만 확장해 재측정"은 이 설계 덕분이다.

## 실행 예

```bash
# 10K 재라벨 결과에 대한 앵커 검증
python -m ner.augmenters.wikiann_vi.wikidata_anchor \
    --input data/wikiann_vi/gemma_8type_full.jsonl \
    --cache data/wikiann_vi/wikidata_cache.json \
    --json-out data/wikiann_vi/anchor_gemma_full.json
```

표준 출력 예:

```
=== Wikidata Anchor Result ===
Entities           : 11629
Unique surfaces    : 7647
With Wikidata Q-ID : 9184
With P31 claims    : 9134
Mapped to 5-type   : 6858
Agreement          : 6524 / 6858 = 0.9512
Per-type agreement:
  PER: 3017/3068 = 0.9836
  LOC: ...
Top unmapped P31 Q-IDs (to consider for table expansion):
  Q190752: 23
  ...
```

## 설정 상수

| 상수 | 값 | 의미 |
|---|---|---|
| `_VI_WIKI_API` | `https://vi.wikipedia.org/w/api.php` | 베트남어 위키 엔드포인트 |
| `_WIKIDATA_API` | `https://www.wikidata.org/w/api.php` | Wikidata 엔드포인트 |
| `_UA` | `ner_pipeline/issue-13 ...` | User-Agent (Wikimedia 권장) |
| `_BATCH` | 50 | 1 요청당 최대 제목/Q-ID 수 |
| `_SLEEP` | 0.2s | 배치 간 대기 |

## 한계와 주의

1. **커버리지 50%** — Wikipedia 등재가 없는 개별 인물명·세부 지명·제품 인스턴스는 검증 불가. 따라서 "앵커 일치율"은 매핑 가능한 부분집합의 지표이지 전체 품질은 아니다.
2. **수작업 매핑 테이블** — `WIKIDATA_TO_CANONICAL`(약 163종)은 사람이 고른 큐레이션. 누락된 세부 타입(예: `Q22806 national library`가 없으면 publisher 류로 잘못 매핑되는 식)이 체계적 에러를 만든다. 리포트 §5.4 카테고리 B 참조.
3. **리다이렉트 왜곡** — Wikipedia 리다이렉트가 다른 개념으로 연결되면 엉뚱한 Q-ID가 할당된다. 예: `'Her Morning Elegance'`(노래) → `'Oren Lavie'`(가수) 리다이렉트로 anchor 타입이 PER으로 잡힘.
4. **P31 복수 해석** — 역사적 정치체가 `city` 와 `former country` 를 모두 가지면 테이블 스캔 순서에 따라 타입이 갈린다(첫 매칭 규칙). 진짜 모호 케이스는 불일치로 기록되지만 실제로는 양쪽 모두 타당할 수 있다.
5. **네트워크 의존** — 1K 엔티티 전체 조회는 신규 실행 시 수 분 소요. 실패는 `reason='network'` 로 소프트 폴백되며 재실행으로 채운다.

## 관련 문서

- 스펙: `docs/manual/data/vietnamese-ner.md` §7 (silver 검증 전략 — 두 번째 검증 레이어)
- 이슈: `docs/issues/issue-10-vi-ner-8type-relabel.md`
- 리포트: `docs/reports/vietnamese-ner-schema-expansion-2026-04.md` §5, §5.5
