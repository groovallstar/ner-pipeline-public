# issue-111: 한국어 canonical NER 5종 완성 — PROD/EVT gold 증분

- Issue: https://github.com/groovallstar/ner_pipeline/issues/111
- PR: <!-- 머지 직전 채움 -->
- 브랜치: `feat/issue-111-ko-prod-evt-gold`
- 승인일: 2026-06-16

## 목적

issue #109의 KLUE 유래 4종(PER/LOC/ORG/DAT) seed gold에 PROD·EVT를 LLM
재라벨로 증분해 canonical **NER 5종** gold를 완성한다. PROD·EVT 라벨 > 0,
회색지대 기준(canonical §3.1~§3.3) 적용.

## 범위
- 포함: ko 라벨러 PROD/EVT 6종 전환, KLUE 문장 LLM 재라벨→비-overlap 병합,
  5종 분포·회색지대 검증
- 제외: PII 5종(EMAIL/PHONE/ID_NUM/CREDIT_CARD — PII 주입 후속), ko 분류기
  학습·벤치마크(canonical 10종 완성 후)

## 성공 기준
- [ ] 테스트: `uv run pytest tests/ner`
- [ ] 문서 갱신: canonical-entity-schema(ko 포함), korean-ner-datasets, ko AGENTS
- [ ] 메트릭/검증: ko gold가 PER/LOC/ORG/PROD/EVT 5종 포함(PROD·EVT > 0),
  회색지대 spot-check 통과

## 위험·의존성
- vLLM 백엔드(gemma-4-31B `:8081`) 가용성 — GPU 공유(타 모델 로드 중)
- 워크트리 `/tmp/ner_pipeline-ko-canonical`: 세션 `PYTHONPATH`가 메인 src를
  가리킴 → 모든 명령 `env -u VIRTUAL_ENV -u PYTHONPATH` 필요. gold는 심링크

## 현 상태 (fact)
- base = `origin/develop`(6fa3c87). 109 코드(ko 4종 라벨러 + gold 스크립트)는
  develop에만 존재, issue-108 브랜치엔 없음
- seed gold `data/klue/origin.jsonl`: 26008 레코드 / PER 18871·ORG 10673·
  DAT 10341·LOC 8312 (`label` 필드)
- full relabel(gemma, 26008): PROD/EVT 4944 span(PROD 3689·EVT 1255),
  4093 레코드. 파일럿 정밀도 PROD ~93%·EVT ~90%
- merge(containment-replace) → `data/klue/origin_5type.jsonl`:
  **PER 18529·ORG 10441·DAT 10071·LOC 7964·PROD 3313·EVT 1191** (51509).
  added 4504(replace 971), skipped 440, KLUE override 1192
  (LOC 348·PER 342·DAT 270·ORG 232)
- 검증: overlap 0(flat, BIO 가능)·offset mismatch 0·회색지대 leakage 없음
  (운영리그 bare=ORG 유지, 특정 연도판만 EVT)·`tests/ner` 313 pass

## 결정 로그 (append-only)
- 2026-06-16: base를 origin/develop로 — 109 코드가 develop에만 존재
- 2026-06-16: 백엔드 gemma-4-31B `:8081`(canonical relabel 검증 모델),
  롤아웃 pilot(500)→full(26k)
- 2026-06-16: 파일럿 후 프롬프트 보강(온라인 서비스·웹사이트·named 프로젝트·
  지시어 비-entity) + post-filter(단독 숫자·단일 글자 PROD 드롭) → 재파일럿에서
  PROD 정밀도 87→93%. 게이트 통과 후 full
- 2026-06-16: 병합 정책 additive→**containment-replace** 전환. 근거: flat BIO
  분류기는 overlap span 표현 불가(`data_utils._bio_labels_from_offsets` 토큰당
  1라벨). dry-run에서 additive가 outermost PROD/EVT 30% 손실(`쓰촨 대지진`류)
  확인. outermost(완전포함)만 replace, exact 동일-span 불일치·subset·partial은
  KLUE 유지(보수)
- 2026-06-16: 최종 gold는 `origin.jsonl`을 5종으로 **in-place 승격**(단일 gold).
  4종 seed는 `klue_to_canonical_gold`로 재생 가능, relabel 산출물
  `results/classifier/ko/prod_evt_relabel.jsonl` 보존 → merge in-place 재현 가능

## 미해결 질문 (open question)
- (없음 — PII 5종 증분은 별도 후속 이슈)

## 구현 단계 (가변, 선택)
- [x] 1. ko 라벨러 프롬프트 PROD/EVT 추가 (canonical §2.2/§3/§3.1~§3.3)
- [x] 2. relabel+merge 스크립트 `scripts/ko_prod_evt_relabel.py`
- [x] 3. relabel 실행 (full 26k, gemma `:8081`) + containment-replace 병합
- [x] 4. 검증 — 5종 분포·회색지대 spot-check·flat(overlap 0)·pytest 313 pass
- [x] 5. docs 갱신 (canonical-entity-schema ko 포함, korean-ner-datasets, ko AGENTS)

---

## 변경 요약
- ko 라벨러(`ner_prompts.py`) 6종 전환: PROD/EVT 정의·규칙·예시 + 기존 "사건/
  작품/선박 제외" 규칙을 PROD/EVT로 재작성 (canonical §2.2/§3/§3.1~§3.3
  회색지대). SINGLE+SYSTEM 프롬프트 정렬
- `scripts/ko_prod_evt_relabel.py` 신규: relabel(vLLM PROD/EVT 수확,
  resume·post-filter) + merge(containment-replace, flat 보장)
- gold `data/klue/origin.jsonl` 4종→5종 in-place 승격
- docs: canonical-entity-schema(ko NER 5종)·korean-ner-datasets·ko AGENTS

## 검증
- 테스트: `uv run pytest tests/ner` → 313 passed, ruff clean
- gold: 26008 레코드, **overlap 0(flat, BIO 가능)**, offset mismatch 0
- 분포: PER 18529·ORG 10441·DAT 10071·LOC 7964·**PROD 3313·EVT 1191**
- 품질: 파일럿 정밀도 PROD ~93%·EVT ~90%, 회색지대 leakage 없음
  (bare 운영리그=ORG, 특정 연도판만 EVT)
- 병합: replace 971(outermost)·skip 440(exact/subset/partial 보수)·
  KLUE override 1192(LOC 348·PER 342·DAT 270·ORG 232)

## 관련 커밋
- `075300b`: ko 라벨러 6종 전환 + relabel/merge 스크립트 + gold 5종 승격 + docs

## 후속 작업
- PII 4종(EMAIL/PHONE/ID_NUM/CREDIT_CARD) 증분 — PII 주입 별도 이슈
- ko 분류기 학습·벤치마크 — canonical 10종 완성 후
