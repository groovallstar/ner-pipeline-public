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

### ontonotes_en/
OntoNotes5(영문) → canonical 10 종 평면 변환. 원본은 사전 토큰화된 18 타입
BIO 라, 자연문을 먼저 복원해야 이 저장소의 공용 통화인 char-offset span 이
나온다. **EN 은 원천이 넘치는 유일한 언어다** — ja·vi·ko 는 원천이 빈약해
LLM 재라벨로 부족한 타입을 만들어냈지만, OntoNotes 는 `PRODUCT`·
`WORK_OF_ART`·`EVENT` 를 이미 갖고 있어 재라벨할 대상이 없다. 그래서 이
모듈은 "지어내기" 가 아니라 "골라내고 옮기기" 다.

#### 주요 파일

| 파일 | 역할 |
|------|------|
| `mapping.py` | 원본 18 타입 → canonical 매핑표 + **전수성 게이트**. 선언되지 않은 타입을 만나면 `UndeclaredTypeError` 로 즉시 중단한다 — 조용히 드롭하면 그 타입이 통째로 빠진 것을 아무도 못 본다 |
| `detokenize.py` | 토큰 배열 → 자연문 복원 + 토큰별 char-offset. 규칙은 코퍼스 76,714 문장 전수 측정에서 나왔다(`-` 는 97.6% 가 단어 사이라 붙이고, `&` 는 `Fleet & Leasing` 이라 띄운다) |
| `convert.py` | BIO 디코드 · 레코드 변환 · **엔티티↔원본 토큰 대조** · 문장 동일 행의 `orig` 묶기 |
| `restore_groups.py` | 주입이 버린 `orig`·`split` 되돌리기 — 아래 |
| `__main__.py` | CLI (`python -m ner.augmenters.ontonotes_en`). split 별로 파일을 갈라 쓴다 |
| `merge_splits.py` | split 파일 셋 → 학습용 단일 JSONL. `split` 필드를 떨어뜨린다 — 아래 |

#### 병합이 `split` 을 떨어뜨리는 이유
`classifier` 는 파일 하나를 받아 스스로 쪼개므로(`--data`) ko·ja·vi 와 같은
자로 재려면 en 도 한 파일이어야 한다. 그런데 세 파일을 그대로 이어붙이면
`split` 이 한 파일 안에서 세 값을 갖고, `validate_group_key` 는 값이 적으면서
선언한 키의 그룹을 가르지 않는 필드를 "더 강한 그룹 키" 로 보고 거부한다 —
같은 `orig` 의 행은 언제나 같은 split 에 있어(경계를 하나도 가르지 않는다)
정확히 그 모양이 되고, `--group-key orig` 가 통째로 막힌다. 재분할이 전제라
원 split 은 학습에 쓰이지 않으며, 어느 split 이었는지는 `id`·`orig` 접두사
(`en-train-`/`en-valid-`/`en-test-`)에 남아 정보도 잃지 않는다.

#### LOC/ORG 경계 — EN 은 JA·VI 관례
`FAC`(공항·역·경기장·다리·고속도로)를 `ORG` 로 흡수한다. KO 의 narrow-ORG
(인공 시설을 아예 버림)를 따르지 않는다. 물려받은 기본값이 아니라 선택이므로
`mapping.py` 모듈 docstring 에 선언해 둔다.

#### 수식 위치 GPE 는 빼지 않는다 — 검토했다가 접은 안

`U.S. troops` 의 `U.S.` 를 §2.3 "국적·소속 수식 명사구 → 비-LOC" 로 보고
제거하는 안을 구현했다가 **표본 검수에서 반박돼 되돌렸다.** 규정 원문만 받은
독립 판정에서 제거 대상 70 건 중 비-entity 판정이 **0 건**이었고(69 LOC·1 ORG),
같은 판정자가 `NORP`·`LANGUAGE`(`Latin American`·`German`) 60 건은 **97%** 를
비-entity 로 봤다. §2.3 이 배제하는 것은 지명에 접미가 붙은 파생어이고 지명의
수식 용법은 다르다는 뜻이다.

JA 실데이터도 같은 방향을 가리킨다 — `data/stockmark/origin.jsonl` 의 LOC
2,899 건 중 국적수식 복합 형태는 **0 건**이다. 일본어는 `日本企業` 가 한
토큰이라 애초에 span 이 안 생긴다. 즉 이 필터를 EN 에만 걸면 "JA 와 일관" 이
아니라 **EN 만 JA 실데이터보다 엄격**해진다.

관련 원칙이 하나 더 있다 — 이슈 #82 는 "장소 자체를 지칭하는 단독·조사
결합만 LOC" 이라고 적었고, 그것만 보면 필터가 맞다. **어느 쪽이든 기존 규칙의
자동 귀결이 아니라 새 결정이며**, 이 저장소는 빼지 않는 쪽으로 정했다.

#### 주입 뒤에는 `orig` 를 되돌려야 한다

`pii/schema.py` 의 `Record` 는 `text`·`entities`·`id` 세 필드만 담는다. 그래서
주입을 지나면 변환이 심어둔 `orig`(형제 묶음 키)와 `split` 이 **사라진다.**
JA 는 `--group-key id` 를 써서 겪지 않는 문제지만 EN 은 `orig` 를 쓴다 —
OntoNotes 가 같은 문장을 여러 번 담기 때문이다.

주입은 한 입력에서 최대 한 행을 내므로 `id` 가 산출물에서 그대로 유일하고,
`restore_groups` 가 그것으로 이어 붙인다. 주입이 문장을 다시 써서 `text` 는
갈라지지만 `orig` 는 원문으로 묶으므로 그룹 보호가 유지된다. 실측 행 수는
실행마다 달라지므로 여기 적지 않는다 — 최신 값은 `docs/issues/issue-209-*.md`
§현 상태 에 둔다.

```bash
python -m ner.augmenters.ontonotes_en.restore_groups \
    --source-dir data/ontonotes_en --injected-dir data/ontonotes_en/pii
```

이 단계를 빠뜨리면 후속 학습에서 `--group-key` 를 줄 수단이 없다. 전량
불변식 테스트가 산출물에 `orig`·`split` 이 있는지와 `validate_group_key`
통과를 함께 본다.

#### 물려받은 중복 문장
OntoNotes 공식 split 은 같은 문장을 여러 번 담는다(train 59,924 행 중 고유
55,154). 방송·전화 대화의 `yeah`·`Uh-huh.` 같은 짧은 발화가 대부분이라 외울
엔티티가 없지만, split 을 가로지르는 것 중 엔티티를 가진 것이 있다 — test
span 8,244 중 **67 개(0.81%)** 가 train 에도 나타난다. 제거하지 않되, **근거를 정정한다.** 처음에는 "제거하면 외부 공개 수치와
비교가 깨진다" 를 이유로 들었으나 그 비교 가능성은 이미 없다 — 18 종을 6 종으로
줄인 시점에 라벨 공간이 달라져 published OntoNotes NER F1 과 나란히 놓을 수
없다. 남은 이유는 둘이다: ① 공식 split 을 그대로 쓰는 것이 재현·인용에
유리하고, ② 노출 규모가 test span 의 0.81% 로 작다. **규모가
고정된 것과 유지 근거가 유효한 것은 다른 문제이므로**, 이 수치가 커지면 유지
결정을 다시 봐야 한다. 테스트가 못 박는 것은 규모뿐이다.

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
| `generators/en.py` | 영어(**미국 단일**) PII 생성기. SSN 은 미발급 지역번호(`000`·`666`·`900`–`999`), 전화는 NANP 예약 대역(`555-0100`~`555-0199`)만 써 실유효 번호를 만들지 않는다. 로케일 전용 `EMAIL_DOMAINS` 를 선언한다 |

#### 라벨 스키마
- **내부 PII 토큰**(생성·병합 전): `NAME`, `PHONE`, `ADDRESS`, `DAT`,
  `ID_NUM`, `EMAIL`, `CREDIT_CARD` (7종 — `ADDRESS`·`NAME`은 병합 대상)
- **병합 규칙** (`DEFAULT_MERGE_RULES`): `NAME → PER`, `ADDRESS → LOC`
  (둘 다 무조건 병합). 학습 데이터에는 `NAME`·`ADDRESS` 라벨이 존재하지
  않는다
- **최종 출력 라벨**: canonical 10종 평면 (NER 5종 `PER/LOC/ORG/PROD/EVT` + PII 5종 `DAT/EMAIL/PHONE/ID_NUM/CREDIT_CARD`)

#### 언어가 국가를 정하지 않는 경우 — EN
ko·ja·vi 는 언어와 국가가 1:1 이라 전화·ID 체계가 자동으로 정해졌다. 영어는
US·UK·AU 가 전부 달라 **미국 단일 체계로 못 박았다**(원천 OntoNotes5 가 미국
뉴스·방송 중심이라 원문 도메인과도 맞는다). 다른 영어권을 쓰려면 생성기를
새로 선언해야 하며, 조용히 섞으면 평가 해석이 흐려진다.

#### EN 은 `DAT` 을 주입하지 않는다
원본 OntoNotes `DATE` 가 gold 로 주기 때문이다. 주입 대상은 4 종
(`EMAIL`·`PHONE`·`ID_NUM`·`CREDIT_CARD`)이며, KO 가 KLUE 에서 날짜를 받는
것과 같은 구조다.

#### 이메일 도메인은 로케일이 선언하면 그것을 쓴다
공용 `base.EMAIL_DOMAINS` 는 ja·vi·ko 도메인이 섞여 있어 영문 문장에
`docomo.ne.jp` 가 붙는다. 생성기 모듈이 `EMAIL_DOMAINS` 를 선언하면
`generate_pii` 가 그것을 쓴다(additive — 선언하지 않은 ja·vi·ko 는 기존 동작
그대로).

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

**이 플래그가 gold 의 성격을 바꾼다 — 지표를 읽을 때 알아야 한다.** 기본
7종에는 `NAME`·`ADDRESS`·`DAT` 가 들어 있고 앞 둘은 `DEFAULT_MERGE_RULES` 로
`PER`·`LOC` 에 무조건 병합된다. 그래서 플래그 없이 돌린 JA·VI 산출물은
`PER`·`LOC` 의 **1/4~1/3 이 합성분**이고(실측 JA 25.4%·31.3%, VI 32.7%·32.4%)
`DAT` 도 전량 합성이다. 4종만 넣은 KO·EN 은 그 셋이 원천 gold 그대로다
(합성 매처로 재면 1% 미만이고 그마저 오탐이다).

**그러므로 JA·VI 의 `PER`·`LOC`·`DAT` 점수를 사람 gold 성능으로 읽으면 안
된다.** 합성 PII 는 정규 패턴이라 모델이 쉽게 맞히므로 그 세 지표가 위로
당겨진다. 언어끼리 그 라벨을 나란히 놓는 비교도 성립하지 않는다 — KO·EN 은
같은 라벨이 순수하다.

**`--lang` 은 주입 세트를 바꾸지 않는다.** 언어별 기본값 분기가 없어
`--lang ko` 를 줘도 플래그를 빼면 7종이 들어간다. 재생성 때 빠뜨리면 어떻게
되는지는 언어마다 갈린다 — EN 은
`tests/ner/augmenters/ontonotes_en/test_corpus_invariants.py` 의
`test_injection_replays_exactly` 가 4종을 못 박은 채 주입을 재생해 산출물과
대조하므로, 합성 `PER`·`LOC`·`DAT` span 이 재생 불가로 잡혀 시끄럽게 깨진다
(주입 산출물이 있을 때만 도는 검사다). **KO 는 그 대조가 없어 gold 의 성격이
조용히 뒤집힌다** — 플래그를 쓰는 두 언어 중 무방비인 쪽은 KO 하나다.
JA·VI 는 기본 7종이 의도된 상태라 이 방향으로 뒤집힐 것이 없지만, 그쪽도
동형 재생 대조는 없다.

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
