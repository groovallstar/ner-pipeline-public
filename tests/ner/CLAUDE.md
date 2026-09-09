# tests/ner/ — 테스트 가이드

pytest 기반 단위·통합 테스트. augmenters(pii, wikiann_vi), labelers(ja, ko, vi),
llm_eval, metrics, validity 등 NER 파이프라인 전반을 커버한다.

## 실행

```bash
pytest tests/ -v        # 전체
ruff check              # 린트
```

## 구조

```
tests/ner/
├── CLAUDE.md
├── test_base_labelers.py
├── test_llm_helpers.py
├── test_span_f1.py
├── test_span_matcher.py
├── test_span_metrics.py
├── test_validity.py   # ner.validity — σ_fold·σ_repro, comparability,
│                      # 누출 근거, compare verdict 우선순위
├── augmenters/
│   ├── ontonotes_en/  # OntoNotes5 → canonical 변환·매핑 전수성, 자연문 복원,
│   │                  # split 병합·group 복원, FAC 판정 표와 디스크 이관 원장
│   ├── pii/           # suffix·llm injector, label merger, loader, verifier,
│   │                  # ko·vi·en 생성기·주입, __main__ verify 라벨러 lang 분기
│   └── wikiann_vi/    # 재라벨 파서·kappa·Wikidata anchor·confidence 병합,
│                      # silver-갭 additive 삽입
├── classifier/        # confidence_threshold, boundary weights, data_utils,
│   │                  # encode, error_analysis, kfold_pool,
│   │                  # ko·en 배포 패키지, lang 배선
├── golden/
│   ├── ontonotes_en/  # EN 변환 산출물의 byte-고정 스냅샷
│   └── validity/      # verdict 별 byte-고정 스냅샷 11종 (pass·fail·invalid·
│                      # inconclusive) — compare() 출력 전체를 대조
├── labelers/
│   ├── test_ko_evt_r2_audit.py  # KO EVT canonical 규칙 감사·R2 회수 —
│   │                  # 형태소 경계, 위반 탐지(축2·축3 트림·R1)와 겹침 가드,
│   │                  # 회수 삽입·불변 타입 byte-identical, 보고 전용 bare 비대칭
│   ├── test_ko_evt_axis1_audit.py  # KO EVT 축1-복합 head 열거·자리 스캔·회수 —
│   │                  # 열거의 결정성(span 중간에 묻힌 head 포착·빈도 문턱·
│   │                  # 고유명 이력으로 수식부 제외·재현 파라미터),
│   │                  # 고유명 가드(1글자 성씨·저비율 제외·조사 붙은 선행어절
│   │                  # 제외), gold EVT 겹침을 버리지 않고 분류,
│   │                  # canonical 파싱 ↔ 모듈 상수 동기(head·고유명 타입·가드
│   │                  # 임계·사유코드), 배정 원장 전수 커버,
│   │                  # 적용(원장↔현 규칙 경계 합의·row_id 해석·기존 span 위 삽입
│   │                  # 거부·불변 타입 무영향), 경계 잔여 계수기(과축소 검출 ·
│   │                  # 라벨된 고유명은 위반 아님 · head 목록과 무관한 모집단),
│   │                  # FP 사각 진단(모양 판정이 head 목록을 안 쓴다·낱말 조각 분리),
│   │                  # 비순환 재채점(동일 팔 Δ 0·회수 행 제외·행/분할 불일치 거부),
│   │                  # 커밋된 산출물 4종의 지문 대조(사전등록이 순서를 집행한다)
│   ├── test_ko_evt_holiday_audit.py  # KO 명절·기념일·절기 이름의 EVT 판정 —
│   │                  # 후보 두 그물이 각자 무엇을 잡는지(어근은 접미 없는
│   │                  # 이름·접미는 어근이 모르는 이름·숫자 날짜는 제외),
│   │                  # 게이트가 막는 것(판정 표면형의 라벨 이동·부분 적용·
│   │                  # span 삭제·자리 이동·미배정·유령 판정·EVT_KEEP 강등),
│   │                  # **게이트가 못 보는 것도 고정한다** — 후보 그물 밖 span 은
│   │                  # 통과하며 그 한계를 산문이 아니라 테스트가 못 박는다,
│   │                  # 완전성 앵커(표면형 지문이 재라벨에 불변·후보 밖까지
│   │                  # 덮는다·새 표면형이면 차단), canonical §5.3 ↔ 모듈 상수
│   │                  # 양방향 동기와 마커 결손 시 시끄러운 실패,
│   │                  # 표기 변이 회귀(핼러윈·할로윈을 넣고 핼로윈을 빠뜨린
│   │                  # 실패가 실제로 났다), 원장이 자기 검사를 끄지 못함
│   │                  # (sites 필수·중복 거부·count 자기모순 거부·검증한 자리
│   │                  # 수를 리포트에 남김), 재라벨(자리 집합 등식·다른 gold
│   │                  # 거부·DAT 아닌 계획 자리 거부·라벨 외 필드 보존),
│   │                  # **커밋된 산출물 ↔ 실제 gold 반대심문** — gold 가
│   │                  # gitignore 라 diff 에 안 남아 산출물이 유일한 기록인데,
│   │                  # 원장 되돌리기가 재라벨 전 gold 를 byte 로 재현하므로
│   │                  # provenance 의 before 지문이 저장소만으로 반증된다,
│   │                  # **기간 머리가 뒤따르는 자리** — canonical 의 기간 머리
│   │                  # 목록이 판정에 실제로 쓰이는지(안 쓰이면 파싱·동기만 되고
│   │                  # 세는 코드가 없다), 그런 자리를 원장이 따로 판정하지
│   │                  # 않으면 차단, 자리별 예외가 그 한 자리만 DAT 로 남김,
│   │                  # 등위로 머리를 나눠 갖는 자리까지 따라가는지(바로 뒤만
│   │                  # 보면 한 명사구 안에서 타입이 갈린다),
│   │                  # 커밋된 원장의 예외 15 자리(인접 11 · 등위 4) = 기계로
│   │                  # 다시 찾은 자리이고 attachment 도 홉 여부와 대조된다,
│   │                  # **기간 머리가 아닌 머리**(범주·하루) — 넓은 스윕이
│   │                  # 게이트가 강제하는 집합보다 커야 다음 머리 부류가
│   │                  # 드러난다(자기 탐지기로 자기 완전성을 재면 0 이 나온다),
│   │                  # 미검토 자리는 차단되고 reviewed_heads 가 이유를 남긴다,
│   │                  # 무회귀 산출물의 자기선언 대신 다시 세어 대조
│   ├── test_ko_locorg_ledger.py  # canonical §2.5 ↔ KO gold 원장 —
│   │                  # 원장 자체 정합성(gold 없이도 돈다), 모집단이 §2.5
│   │                  # 예시에서 나오는지, gold 지문 불일치는 거부,
│   │                  # `data/` 없는 워크트리에서는 경고를 남기고 skip
│   ├── test_ko_split_audit.py  # 옛 두 팔의 fold 대응 사후 감사 —
│   │                  # 회수 역적용(좌표 일치분만·판정별 필드명 분기),
│   │                  # 겹침의 자가 Jaccard 가 아님, 문턱이 실측 전에 고정됨,
│   │                  # 근거 없이 갈리는 시나리오에서 판정 거부(추측을 결정으로
│   │                  # 굳히지 않는다), 양성 대조·역산 지문 실패 검출,
│   │                  # 커밋된 산출물의 자기 정합성
│   ├── test_prompt_token_budget.py  # ko·ja·vi 고정 지시문이 컨텍스트 예산
│   │                  # 안인지 — 프롬프트 + 출력 예약이 max-model-len 을 넘으면
│   │                  # vLLM 이 전건 400 으로 거절하고 리포트는 정상 생성돼
│   │                  # 모델 성능 저하로 읽힌다. 토크나이저는 로컬 HF 캐시
│   ├── test_en_labeler.py  # EN 라벨러 — PII 주입 교차 검증 전용 경로
│   ├── test_canonical_scope_markers.py  # canonical 절의 적용 언어 마커
│   ├── test_canonical_section_refs.py   # canonical 절 참조의 실재
│   ├── ja/            # test_ja_dataset_loader (canonical JSONL 로딩)
│   └── vi/            # test_dataset_loader (VI canonical JSONL 로딩)
├── llm_eval/          # eval_mode CLI dispatch, vi_silver_quality,
│                      # wikiann_vi_gold 테스트
└── scripts/           # 배포 패키지 포장(build_ner_prod), EN 배포 추론
```

`labelers/` 에 `ko/` 디렉토리는 없다 — 한국어 라벨러 테스트는 최상위
`test_base_labelers.py` 와 `labelers/test_ko_*.py` 에 흩어져 있다. `test_ko_ner_prompts.py` 는 축1 head 동기에 더해 canonical §2.5 ↔
프롬프트 LOC/ORG 동기, canonical §3.3·§5.3 ↔ 프롬프트 명절 판정 동기를
양방향으로 대조한다. 명절 쪽이 묶는 것은 **네 목록**(하루·기간·범주 머리와
구간 이름)과 **예시**다 — gold 는 명절 이름 138 자리를 `EVT` 로 옮겼는데
프롬프트는 잠금 밖이라, 둘이 갈리면 라벨러가 옛 기준으로 답하고 그 하락이
재라벨 탓인지 모델 탓인지 구별되지 않는다. 예시 쪽은 지목 검사(대조쌍의
존재·파생 안의 이름을 따로 안 뽑음·등위 자리)와 전수 검사(명절 어근을 담은
모든 예시 span 을 `following_span_head()` 에 태워 기대 타입 계산)를 함께 둔다 —
지목만 두면 **새 예시가 `추석`=DAT 를 가르쳐도 지나간다.**

`tests/server/`는 별도 패키지(server 계약·통합·live 테스트)로, 본 문서는
`tests/ner/`만 다룬다. 상세 전략: `src/server/CLAUDE.md`.

## 테스트 전략

- 단위 테스트: 개별 labeler·span_matcher·metrics 함수 (LLM API 호출은 mock)
- 통합 테스트: 데이터셋 로딩 → 라벨링 → 평가 파이프라인 검증
- 벤치마크: `python -m ner.llm_eval` CLI로 실행 (tests/ 외부)
- 새 테스트 추가 시 기존 패턴 참고 (`mock AsyncOpenAI`, fixture 공유)

## 주의사항

- LLM 백엔드(vLLM) 연동 테스트는 서버가 실행 중이어야 함
- 조건부 skip 은 GPU 유무가 아니라 **로컬 자원 부재**로 걸린다 — HF 캐시가 없으면
  `skipif`(`labelers/test_prompt_token_budget.py` — 토크나이저 캐시),
  선택적 패키지(`pyvi`·`fugashi` 등)가 없으면
  `importorskip`(`classifier/test_encode.py`). CI 설정은 리포에 없어(호스트 실행 전제)
  "CI 에서 스킵" 이라는 경로 자체가 없다
- 테스트는 `PYTHONPATH` 설정 없이 동작 (uv editable install 기준)
- 저장소 루트는 `pyproject.toml` 의 `[tool.pytest.ini_options] pythonpath` 가
  올린다 — 형제 테스트 모듈을 `tests.ner...` 로 읽는 파일이 있어서다. 환경변수
  `PYTHONPATH` 가 아니라 pytest 세션 안에서만 사는 경로라 위 금지와 다른 축이고,
  없으면 `pytest` 와 `python -m pytest` 의 결과가 갈린다

## 의존성

- 내부: `ner.metrics.{bio_metrics,span_metrics}`, `ner.validity.*`,
  `ner.labelers.{ja,vi}.*`, `ner.classifier.*`, `ner.llm_eval.*`,
  `ner.augmenters.{pii,wikiann_vi}.*`
- 외부: `pytest`, `ruff`
