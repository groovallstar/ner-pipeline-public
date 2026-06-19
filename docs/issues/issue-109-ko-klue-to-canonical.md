# issue-109 한국어 라벨러 canonical 정렬 (KLUE 4종 개명 + TI/QT 드롭)

## 목적

ja·vi는 canonical 10종 평면 스키마를 공유해 분류기·메트릭·교차비교를
함께 쓰지만, 한국어는 KLUE 6종(`PS/LC/OG/DT/TI/QT`) 독자 스키마라 합류
불가. 한국어 라벨러·gold를 canonical 명칭으로 정렬하는 첫걸음.

## 범위

- **이번 이슈**: KLUE 실재 4종 개명(`PS→PER, LC→LOC, OG→ORG, DT→DAT`)
  + `TI`(시간)·`QT`(수량) 드롭 — canonical에 대응 타입 없음.
- **범위 외(후속 이슈)**: PROD/EVT·PII 6종 gold 구축(KLUE에 라벨 없음),
  ko 분류기 학습, canonical 스키마 확장(TIME/QUANTITY 추가) 여부.

### 운영 원칙

- **gold는 점진적으로 누적**한다: KLUE 유래 4종(PER/LOC/ORG/DAT)을 seed로
  먼저 만들고, PROD/EVT·PII 6종을 후속 증분(LLM 재라벨·합성주입)에서 같은
  gold에 얹어 canonical 10종으로 키운다.
- **canonical 10종 완성 전에는 벤치마크/학습을 돌리지 않는다.** 따라서 본
  이슈에 "ko 벤치마크 정합"은 포함하지 않는다(중간 수치는 무의미).

## 설계 메모 (현 구조 분석)

- ko 스키마 원천: `labelers/ko/ner_prompts.py`
  (`DEFAULT_ENTITY_TYPES` + SINGLE/SYSTEM 프롬프트·few-shot 예시).
- `tag_aligner._TAG_NORMALIZE_MAP_KO`가 canonical→KLUE(`PER→PS`) 역방향
  이고 **ja가 동일 맵을 공유**(`_TAG_NORMALIZE_MAPS["ja"]`).
  → 프롬프트만 바꾸면 normalize가 되돌려 **무효**. ko 맵을 ja와 분리하고
  ko를 canonical 방향으로 전환해야 라벨러 출력이 유지된다.
- gold(KLUE jsonl)는 `PS/LC/OG/DT/TI/QT` → 후속 단위(2)에서 재매핑.

## 분해 (작은 단위 · 체크박스=진행추적)

- [x] 1. 라벨러 프롬프트 canonical 전환 + `tag_aligner` ko/ja normalize
       맵 분리
- [x] 2. KLUE → canonical **seed gold**(4종 char-span JSONL) 생성
       — `labelers/ko/klue_to_canonical_gold.py`
- [ ] 3. docs 갱신 (`canonical-entity-schema`, `korean-ner-datasets`,
       `tag_aligner`, ko AGENTS)

## 단위 1 상세

### 변경

- `labelers/ko/ner_prompts.py`:
  - `DEFAULT_ENTITY_TYPES = ["PER", "LOC", "ORG", "DAT"]`
  - 타입 정의 6→4 (TI·QT 섹션 제거), `PS→PER/LC→LOC/OG→ORG/DT→DAT` 개명
  - few-shot 예시 재작성: QT/TI 스팬 제거, 나머지 타입명 개명
  - 헤더 docstring 갱신 (어노테이션 규칙은 KLUE 가이드라인 유지)
- `labelers/tag_aligner.py`:
  - `_TAG_NORMALIZE_MAP_JA` 신설 = 기존 KO 맵(canonical→KLUE) **그대로
    복사** → ja에 연결 (ja 동작 불변, 회귀 0)
  - `_TAG_NORMALIZE_MAP_KO`를 canonical 방향으로 교체:
    `PS→PER, LC→LOC, OG→ORG, DT→DAT` (+ PERSON→PER 등 영문 변형).
    `PER/LOC/ORG/DAT`는 미존재 시 그대로 통과. TI/QT는 매핑 없이 통과
    (드롭은 단위 2의 gold 재매핑에서 처리)

### 검증

- `uv run pytest tests/ner` 통과 (회귀 0, 특히 **ja**)
- `normalize_tag` 직접 점검: ko(`B-PS→B-PER`, `B-PER→B-PER`),
  ja(`B-PER→B-PS` 유지)
- 프롬프트 렌더 + `DEFAULT_ENTITY_TYPES == ["PER","LOC","ORG","DAT"]`
- `ruff check` 통과

## 단위 2 상세

### 변경

- `labelers/ko/klue_to_canonical_gold.py` 신설: KLUE 음절 BIO →
  canonical char-span gold JSONL.
  - `PS→PER, LC→LOC, OG→ORG, DT→DAT` 재매핑, TI/QT 스팬 드롭
  - 음절 토큰 재구성 텍스트 기준 char offset(start/end) 부여
  - 출력 계약 = 분류기 `data_utils.load_jsonl`
    (`{id, text, entities:[{label, start_char, end_char, text}]}`)
  - 출력: 프로젝트 루트 `data/ko_klue/klue_canonical.jsonl` (gitignore)

### 검증 (벤치마크 없이)

- 변환: train 21008 + validation 5000 = **26008 레코드, 48197 엔티티**
- `data_utils.load_jsonl` 로드 성공(라벨 전부 canonical 10종 내)
- 라벨 분포: PER 18871 / ORG 10673 / DAT 10341 / LOC 8312 — **4종만**
- **TI/QT 부재, KLUE 약어(PS/LC/OG/DT) 부재** 확인
- char offset 자기일관성(전수 26008): `text[start:end] == entity.text`
  불일치 **0**
- 독립 토큰-조인 참조와 전수 대조: 불일치 **0** (변환 정확성 교차검증)
- `ruff check` 통과

### 공백 경계 절단 수정 (refuter 발견)

초기 변환기가 공백 토큰을 무조건 엔티티 경계로 처리해, KLUE가 공백을
`I-`로 태깅한 다어절 엔티티(예: "5공화국 시절", "삼성물산 건설부문
대학생기자단")의 꼬리를 절단했다(~15%, 7328 엔티티). 공백 토큰이 자체
BIO 태그를 따르도록 수정 → 다어절 7328건 복원, 토큰-조인 참조와 전수
0 불일치로 확인.

## 구현 결과 (단위 1·2)

- 단위 1·2 코드·검증 완료. gold는 canonical 4종 seed 확보.
- 다음: PROD/EVT·PII 6종 증분(후속 이슈) → 10종 완성 후 분류기·벤치마크.
