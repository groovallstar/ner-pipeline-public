# issue-116 — VI PII 자연 주입: suffix 스캐폴딩 제거 + 가설 반증

- GitHub: #116 (area:augmenters, area:classifier)
- 브랜치: `feat/issue-116-vi-pii-natural-injection` (develop 기반)
- 스핀오프: 본 이슈 진행 중 #124(VI NER 원문 cross-fold 누출)가 분리됨 →
  #124 가 자연 코퍼스를 누출-free 로 재학습, 본 이슈는 그 산출물을 재사용.

## 배경 / 가설

VI 학습 코퍼스는 **suffix 모드**로 PII 가 주입돼 있었다 — 문장 끝에 경직된
라벨 접두어를 붙인 스캐폴딩(전체 행의 81.7% 가 `Liên hệ:`·`SĐT:`·`CCCD:`·
`Email:`·`Ngày:`·`Địa chỉ:`·`Thẻ:` 포함):

```
... Đồng bằng sông Cửu Long. Liên hệ: Võ Ngọc Tuấn. Email: x@icloud.com.
```

**가설(Phase 2 잔여)**: "접두어 뒤 = PII" 표면 단서로 모델이 과적합 →
PII 5종 F1 ~0.99 가 과대평가다. 자연 문맥 PII 일반화가 미검증.

## 결정 — 접근

- **NER = 방법1 재배치(원문 NER 재사용)**: v3 정제 NER 을 그대로 쓰고 자연
  주입된 새 문장에서 string-match 로 위치만 재배치. 통째 §3 재라벨링은 NER
  합의 노이즈를 재유입하므로 비채택(보존율 임계 미달 시에만 폴백).
- **LLM verify 제거**: 방법1 에선 NER(재사용)·PII(생성값 exact-match) 라벨이
  *구성상 정답*이라 독립 교차검증이 무가치. 게다가 기본 `drop_span` 은 모델이
  못 알아본 어려운 자연 PII 를 제거해 측정 무결성을 깬다(suffix 0.99 낙관을
  다른 경로로 재현). → 품질 게이트 = **결정론 검사 + 보존율 + 수동 표본**.
- **단일 `pii_all.jsonl` 정책**: 버전 접미사(`_v2`/`_v3`) 폐기.
- **모델**: 주입 = Qwen3.6-35B-A3B-AWQ(8082). NER 원천 = v3 merge.

## 구현

- `scripts/build_vi_natural_source.py`: suffix gold → (원문, 원문 NER) 추출
  (`resilver_vi_isolated.build_source` 경계 복원, 경계 이후 주입 PII 제거).
- `augmenters/pii --mode llm`: 원문에 PII 문중 자연 삽입(`--inject-temperature`
  기본 0.7, 낮출수록 PII verbatim 충실도 ↑) → `pii_all.jsonl`(37,706 행).
- `scripts/validate_vi_natural.py`: 결정론 게이트(마커·offset round-trip·드롭율·
  원문 NER 보존율·무라벨 email/phone 스캔).

## 측정 결과

### 1. 배포 코퍼스 결정론 게이트 (37,706 행 직접 재검증)

| 지표 | 값 | 기대 |
|---|---|---|
| suffix 마커 행 | **0** | 0 |
| offset round-trip 불일치 | **0** | 0 (100% 정합) |
| 무라벨 email / phone | 0 / 7 | 낮음 |
| NER / PII span | 54,895 / 35,511 | #124 support 와 정확 일치 |

→ 배포 코퍼스 = #124 감사 코퍼스 provenance-lock 확인. 출처:
`results/classifier/vi/audit/natural_gate_deployed.json`.

### 2. 보존율 (주입시점 게이트 — 배포 코퍼스 재측정 불가)

원문 NER 보존율 overall **99.41%**(EVT per-label 92.9% — 저support 619).
v3 source NER 이 단일 `pii_all.jsonl` 정책으로 삭제돼 배포 코퍼스 직접
재측정은 불가 → #124 NER5 F1 무붕괴가 무결성 교차검증.

### 3. suffix v3 → 자연 비교 (같은 leaked 분할, phobert / xlm-r)

micro F1. suffix v3 = `resilver_v3` pooled(leaked, 38,371), 자연 = #124
leaked(37,706). 같은 leaked 분할 *방식*으로 주입 스타일 효과를 **근사** 분리
(코퍼스 크기 −1.7%·내용 차 confound 는 잔여 한계 참조).

| 지표 | suffix v3 leaked | 자연 leaked | Δ(주입방식) |
|---|---|---|---|
| overall | 0.9513 / 0.9525 | 0.9506 / 0.9517 | −0.07 / −0.08 |
| NER 5종 micro | 0.9252 / 0.9274 | 0.9198 / 0.9226 | −0.54 / −0.48 |
| **PII 5종 micro** | 0.9921 / 0.9917 | **0.9985 / 0.9972** | **+0.64 / +0.55** |

**PII per-entity (phobert, suffix v3 leaked → 자연 leaked)** — 전 종목 개선,
suffix 마커가 있던 DAT(`Ngày:`)·CREDIT_CARD(`Thẻ:`)에서 최대:

| ent | suffix v3 → 자연 (Δpp) |
|---|---|
| DAT | 0.9849 → 0.9980 (**+1.31**) |
| CREDIT_CARD | 0.9859 → 0.9987 (**+1.28**) |
| EMAIL | 0.9966 → 0.9999 (+0.33) |
| PHONE | 0.9961 → 0.9987 (+0.26) |
| ID_NUM | 0.9970 → 0.9973 (+0.03) |

출처: `results/classifier/vi/audit/natural_vs_suffix_comparison.json`.

### 4. 배포(누출-free) 수치 = grouped

자연 코퍼스 group-kfold(누출-free)의 정직한 일반화 성능:
overall 0.9437 / 0.9430, NER5 0.9086 / 0.9083, PII5 0.9983 / 0.9974.

## 해석 — 가설 반증 (방향까지)

- **PII 0.99 는 suffix 스캐폴딩 인플레가 아니다.** 자연 주입에서도 PII 5종
  ~0.99 유지, 오히려 **+0.5~0.6pp 높다**. suffix 마커는 PII 를 *부풀린* 게
  아니라 DAT/CC 에서 오히려 약간 *해쳤다*(접두어가 날짜·카드 경계를 흐림).
  → 0.99 는 **PII 내부 포맷(이메일·전화·ID·카드 형태) 학습**의 산물.
- **NER 체계적 회귀 가능성 낮음**: NER5 자연 −0.5pp(붕괴 아님). v3 라벨
  재사용이라 체계적 회귀 가능성은 매우 낮으나(보존율 99.4%로 완벽은 아님),
  잔차는 문중 문맥 난이도·코퍼스 차 효과.

## 잔여 한계

- 보존율 배포 코퍼스 직접 재측정 불가(v3 source 삭제) — 위 §2.
- NER5 자연 −0.5pp 가 (a)문중 문맥 난이도 (b)코퍼스 크기차(38.4k→37.7k) 중
  무엇인지 미분리 — suffix 코퍼스 폐기로 동일분할 재측정 불가.

## 검증 — 수락 기준

- [x] **1. 자연 주입 코퍼스** — suffix 마커 0, offset round-trip 100%, PII
  무변형(배포 37,706 직접 재검증).
- [x] **2. 원문 NER 보존율** — 99.41%(≥97%), EVT per-label 92.9% 기록.
  배포 재측정 불가는 #124 NER F1 무붕괴로 보강.
- [x] **3. 결정론 노이즈 + 표본** — 무라벨 email 0/phone 7. N=50 표본 마커 0,
  PII 가 문법절(`qua địa chỉ …`·`thanh toán qua thẻ …`·`có số định danh là …`)
  에 삽입 — 문말 위치도 스캐폴딩이 아닌 자연 종결(위치 비율은 자연스러움 지표
  아님, 판별은 마커 부재 + 문법 삽입).
- [x] **4. 5-fold×2 재학습 + 비교** — #124 산출물 재사용, suffix v3 대비
  PII/NER per-entity 비교표(위 §3), NER 회귀 없음(무결성).
- [x] **5. 자연 PII 일반화 특성화 + 문서화** — 가설 반증 정량화(PII +0.5~0.6pp,
  과대평가 아님), history Phase 3 + 본 아카이브.
- [x] 수치 ↔ `results/*.json` 정합(refuter 게이트).
