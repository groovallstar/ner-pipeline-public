# issue-169 — 누출 검증의 그룹 키를 데이터가 선언하게

## 배경

합성 PII 주입과 재라벨은 한 원문에서 여러 행을 파생시킨다(**형제 행**). 형제가
train·test 로 갈리면 모델이 학습에서 본 정답을 시험에서 다시 만나 span F1 이
부푼다. 이를 막는 장치가 `--group-key` 이고, 지킨 것을 확인하는 장치가
`kfold_pool` 의 cross-fold 중복 카운터다.

그런데 확인 장치가 두 방향으로 고장나 있었다.

## 문제

### 1. 검사기가 그룹 키를 고를 수 없었다

학습 쪽은 `--group-key` 로 필드명을 받는데 `kfold_pool` 은 `orig` 라는 이름이
박혀 있었다. `data/klue/pii_all.jsonl` 에는 `orig` 가 없어(25,989행 전부) 문장
전체 비교로 물러섰다.

그런데 **한국어 PII 주입은 문장을 재작성한다.** PII 가 없는 5,213행만 원본
KLUE 와 글자가 같고 나머지 20,776행은 새로 지어진 문장이다. 형제끼리 글자가
전부 다르므로 문장 비교는 하나도 잡지 못한다. 카운터는 구조적으로 항상 0 이었고,
`metrics/variance.py` 는 그 0 을 "누출 없음" 으로 읽었다.

### 2. 그룹 키를 잘못 고르면 누출이 초록으로 통과했다

정의-시점 반박자가 찾았다. VI 를 `--group-key id` 로 돌리면 `id` 가 전부
고유(37,706/37,706)라 unit 이 전부 싱글턴이 되어 사실상 행 단위 분할이다.
**형제 3,803 그룹이 fold 를 가로질러 샌다.** 그런데 같은 `id` 로 중복을 세면
0 이라 게이트가 초록을 준다. 고치려던 실패가 다른 이름으로 부활한 셈이다.

### 3. 정직한 opt-out 이 벌받았다

`variance.compare` 는 비교가능성 판정보다 먼저 누출 카운터를 무조건 읽는다.
`none` 에 `null` 을 넣고 예외를 던지면, 정직하게 포기를 선언한 실험은 크래시하고
거짓 0 을 낸 실험은 통과한다. 인센티브가 편법을 가리켰다.

### 4. 3-way 기본 경로는 무방비였다 (실재 버그)

`split_train_valid_test` 에는 `group_key` 인자가 아예 없었다. VI 를 기본값으로
학습하면 test 3,770행 중 **1,133행(30.1%)이 train 과 형제를 공유**했다(valid 는
9.3%). 문장 전체 중복은 0% 라 기존 fallback 이 하나도 잡지 못했다.
`error_analysis` 도 같은 경로를 쓴다.

VI 리포트는 group-kfold 로 재측정해 헤드라인 수치는 안전하지만, "영구 차단"
이라는 서술은 반쪽만 참이었다.

## 조사에서 밝혀진 사실

### `id` 의 뜻은 데이터셋마다 다르다

| 데이터 | 행 수 | `id` 그룹 | `orig` 그룹 | 그룹 키 |
|---|---|---|---|---|
| KO `data/klue/pii_all.jsonl` | 25,989 | 25,989 | 필드 없음 | `id` 또는 `none` (형제 없음) |
| VI `data/wikiann_vi/origin.jsonl` | 37,706 | 37,706 | 29,337 | `orig` |
| JA `data/stockmark/origin.jsonl` | 5,270 | **5,166** | 필드 없음 | `id` |

KO 의 `id` 는 KLUE 원본 인덱스다. 범위 0~26007 = train 21,008 + dev 5,000 이고
19개가 결손이다. 문장이 재작성된 행에서도 자기 `id` 의 원본 엔티티 표면형이
99.73% 남아 있다(무작위 `id` 대조군 0.37%).

코드 주석 3곳이 "id 는 비고유라 검증 기준으로 쓰지 않는다" 고 적고 있었다.
JA 에서는 참이지만 KO·VI 에서는 거짓이고, 더 중요하게는 **인과가 뒤집혀 있다.**
비고유는 그룹 키가 될 수 없는 이유가 아니라 될 수 있는 유일한 이유다. 고유한
키는 아무것도 묶지 못한다.

과거 `issue-69` 는 "JA id 비고유 발견 → pooled 무결성 검사를 id 기준에서 text
기준으로 변경(text 는 전부 고유)" 이라 기록한다. 형제를 묶던 필드를 버리고 전부
고유한 필드로 갈아탄 것이다. 그때부터 JA 누출 검사도 무력했다(10-fold 에서 90개
원문이 fold 를 가로질렀다).

### 고유한 필드는 그룹 키가 될 수 없다

값이 전부 다르면 unit 이 전부 싱글턴이라 그룹 보호가 no-op 이 된다. 그런데 같은
필드로 누출을 세면 중복이 0 이다. **보호는 없는데 게이트는 초록이다.**

## 설계

### 그룹 키는 데이터가 선언하고, 코드가 검증한다

`--group-key` 를 두 분할 경로 모두에서 **필수**로 만든다. 행 단위를 원하면
`none` 을 명시해야 한다 — 미측정 분할이 깨끗한 분할로 읽히면 안 된다.

`validate_group_key` 가 학습 **전에** 세 가지를 본다.

1. 선언한 필드가 모든 행에 있는가
2. 값이 비어 있지 않은가
3. 선언한 키보다 행을 더 강하게 묶는 후보 필드가 없는가

### "더 강하게 묶는다" 의 정의

그룹 수가 적은 것만으로는 부족하다. 후보 필드가 선언한 키의 그룹 경계를
가로지르면(같은 그룹의 행이 서로 다른 후보 값을 가지면) 그것은 형제 표시가
아니라 다른 축의 범주다 — 문서 도메인 같은 것.

따라서 두 조건을 모두 본다.

- 그룹 수가 더 적다
- 선언한 키의 그룹을 쪼개지 않고 통째로 포함한다 (`_coarsens`)

이 두 번째 조건이 없으면, 값이 두 개뿐인 `domain` 필드 하나가 올바른 `orig` 를
밀어낸다. 결과-시점 반박자가 찾은 오탐이다.

한계: `group_key=none` 이면 모든 그룹이 싱글턴이라 두 번째 조건이 항상 참이고,
범주 필드도 후보로 잡힐 수 있다. 이때는 후보가 여럿 보고되므로 사람이 고른다.

### 판정 근거를 함께 기록한다

카운트 0 이 "이상 없음" 인지 "볼 수단이 없었음" 인지 구분해야 한다.
`pooled_metrics.json` 에 `leak_check_basis` 를 남긴다.

| 근거 | 뜻 | 신뢰 |
|---|---|---|
| `group` | 선언된 그룹 키의 값으로 셌다 | ✅ |
| `orig` | 그룹 키 기록이 없는 옛 예측 파일, `orig` 로 셌다 | ✅ |
| `text` | 문장 전체 비교로 셌다 | ❌ 재작성 코퍼스에선 형제를 못 본다 |
| `none` | 행 단위 분할을 명시했다 | ❌ 애초에 측정하지 않았다 |
| `unknown` | 근거 키가 없는 옛 산출물 | ❌ 무엇으로 셌는지 모른다 |

`variance.compare` 는 신뢰 가능한 근거로 센 0 만 통과시키고, 나머지는 크래시가
아니라 `INVALID`(누출 미검증)를 낸다. 정직한 opt-out 이 벌받지 않는다.

다만 **카운터가 0 보다 크면 근거가 약해도 본 것이다.** 약한 근거는 누출을 놓칠
뿐 없는 누출을 만들어내지 않으므로, 이 경우는 `INVALID` 가 아니라 확인된
누출(`FAIL`)이다.

## 구현

| 파일 | 변경 |
|---|---|
| `data_utils.py` | `_build_units` 공용 추출(3곳 중복 제거) / `group_stats` / `stronger_group_keys` + `_coarsens` / `validate_group_key` / `split_train_valid_test(group_key=...)` |
| `__main__.py` | `--group-key` 필수화, `none` opt-out, 분할 전 검증, 3-way 전달, `metrics.json` 에 `group_key`·`n_rows`·`n_groups`, 예측에 `group`·`group_key` |
| `error_analysis.py` | `--group-key` 필수화 + `validate_group_key` |
| `kfold_pool.py` | 그룹 키 인자화(`auto` 기본), `orig` 하드코딩 해제, `leak_check_basis` 기록, `cross_fold_group_dups` |
| `variance.py` | `leakage()` — 카운터 + 근거. 미측정·약한 근거는 `INVALID`, 관측된 누출은 `FAIL` |

## 검증

### 수락 기준

| # | 기준 | 결과 |
|---|---|---|
| 1 | 두 분할 경로 모두 `--group-key` 요구, 미지정 시 중단 | ✅ |
| 2 | 필드 부재·null·더 강한 후보 존재 시 중단 | ✅ |
| 3 | `kfold_pool` 그룹 키 인자화 + 판정 근거 기록 | ✅ |
| 4 | `none` 이면 카운터 `null`, 근거·그룹 수 기록 | ✅ |
| 5 | `variance.compare` 미측정·약한 근거 → `INVALID` (크래시 아님) | ✅ |
| 6 | 무회귀 + 테스트 | ✅ 451 통과, ruff clean |

### 실데이터 판정

```
KO: none=통과 | id=통과      (형제 없음 — 증명적 0)
VI: none=거부 | id=거부 | orig=통과
JA: none=거부 | id=통과
```

### 무회귀

- KO `--group-key id` 10-fold 분할이 행 단위와 **완전히 동일**(그룹 크기 전부 1)
- 3-way `group_key=None` 이 기존 셔플 알고리즘과 bit-for-bit 동일
- 옛 예측 파일로 pooled 재계산 시 strict/relaxed F1 **완전 일치**

### 누출 차단 효과

| 경로 | 이전 | 이후 |
|---|---|---|
| VI 3-way, test 형제가 train 에 | 1,133행 | 0 |
| VI 3-way, test 형제가 valid 에 | 352행 | 0 |
| VI 10-fold `id`, fold 가로지른 원문 그룹 | 3,803 | (거부됨) |
| JA 10-fold, fold 가로지른 원문 | 90 | 0 |

### 반박자

- **정의 시점**: `REVISE`. 고유 키가 거짓 초록을 내는 경로(위 문제 2), 기준
  3↔6 충돌(text fallback 제거가 기존 테스트를 깬다), 인센티브 역전(위 문제 3)을
  구현 전에 찾았다. 기준을 6개로 다시 짜고 진행했다.
- **결과 시점**: `PASS`(`.omc/state/refuter/ab6dd02bee463003.json`). MINOR
  findings 4건 중 3건을 반영했다 — 관측된 누출을 `FAIL` 로, 근거 없는 옛
  산출물을 `unknown`(불신)으로, `error_analysis` 에 검증 추가. 네 번째(범주 필드
  오탐)는 `_coarsens` 로 닫았다.

## 2차 확장 — compare 게이트를 자 기반으로

1차는 **누출**(분할이 새는가)을 닫았다. 2차는 같은 `variance.py` 의 나머지
두 축, **비교 가능성**(같은 자로 쟀나)과 **노이즈 밴드**(Δ 가 우연인가)를
고친다. 둘은 독립이 아니다 — paired 비교는 baseline·candidate 의 fold 멤버십이
같아야 성립하고, 그건 `seed`·`stratify` 가 같아야 하므로, "같은 자" 판정에
그 필드를 넣는 일(Part A)이 paired 밴드(Part B)의 전제조건이다.

이 둘은 **자(채점규칙)를 바꾸는 일**이다. 지금 확정하는 이유: 유효한 실험
결과가 하나도 없다(pii_all.jsonl 데이터만 유효, results/ 는 무의미·진행 중).
소급될 baseline 이 없으므로, 지금 정의를 못 박으면 이후 점수로 자를 고치는
일이 구조적으로 불가능하다 — 자를 바꾸기에 가장 안전한 시점이다.

### 문제 — `check_comparable` 이 파일 경로 문자열을 비교한다

`RULER_FIELDS = (lang, data_path, kfold, group_key)`. 두 방향으로 틀렸다.

- **경로 같고 내용 다름 → 통과**: `data/klue/pii_all.jsonl` 의 ORG gold 를
  10,416→2,797 로 바꿔도 경로가 같아 "같은 자" 로 통과한다(실제 이 저장소에서
  일어난 일). 옛 gold 로 잰 값과 새 gold 로 잰 값이 나란히 놓인다.
- **`seed` 가 자에 없다**: fold 멤버십을 정하는 게 `seed` 인데 RULER 에 없어,
  다른 seed 로 자른 두 run 이 "같은 자" 로 통과한다. paired 비교의 전제가 깨진다.

### 문제 — 노이즈 밴드가 틀린 단위이고 unpaired

Δ 는 pooled per-entity F1(fold 10개 예측을 합친 단일 수)인데, 밴드는 개별
fold F1 의 산포 σ_fold × k 다. pooled 는 개별 fold 보다 안정적이라 밴드가
과하게 넓어 대부분 INCONCLUSIVE 로 떨어진다. 게다가 baseline·candidate 는
같은 fold 를 공유하므로 fold 난이도가 상쇄되는데(paired), 지금은 baseline 의
σ_fold 만 써서 그 검출력을 버린다.

### 설계

**Part A — 내용 지문 + fold 멤버십을 자에 넣는다.**
- `metrics.json` 에 `data_fingerprint`(데이터 내용의 순서 민감 해시: 행별
  text + 정렬된 (label,start,end))와 `stratify` 를 기록한다.
- `RULER_FIELDS = (lang, data_fingerprint, kfold, group_key, seed, stratify)`.
  `data_path` 는 자에서 빠지고 참고용으로만 남는다. 지문이 없는 옛 산출물은
  0/동일 가정 없이 fail-loud("missing" → INVALID).
- 효과: 내용이 바뀌면 지문이 달라 비교 거부, 경로만 바뀌면(rename) 통과.
  gold 를 고쳐 지문이 달라지면 INVALID — 비교하려면 **옛 모델을 새 gold 로
  재채점**(clean 동일-test)해 같은 지문의 baseline 을 만들어야 한다. 자를
  바꾸면 재측정하라는 규율을 코드가 강제한다.

**Part B — paired-fold 방향 일관성 게이트.**

> **정의-시점 반박자가 최초 설계(paired-SEM 밴드)를 부쉈다.** SEM 밴드는
> std(Δᵢ)→0 일 때 0 으로 붕괴해 +0.001 도 real_gain 으로 통과시키고(σ_fold 는
> 실데이터에서 0 이 안 돼 이 버그가 없다), band_k=2 는 n=5 fold 에서 t>2 =
> 단측 5.8%(2.5% 아님)라 anticonservative 하며, k-fold Δᵢ 는 학습셋이 겹쳐
> iid 가 아니라 SEM 이 분산을 과소추정한다. 셋 다 false PASS 방향. 결정적으로,
> σ_fold 가 넓은 것은 **노이즈 게이트에겐 안전한 방향**이고 정확히 재려면
> σ_repro 를 재라는 게 설계된 탈출구다. "paired 가 밴드를 좁힌다"는 틀린
> 목표였다. paired 의 올바른 쓰임은 밴드를 좁히는 게 아니라 **조이는 것**이다.

- 점추정·밴드는 **그대로 둔다**: Δ = pooled per-entity F1(cand−base), 밴드 =
  band_k × σ (σ_repro override 있으면 그것, 없으면 σ_fold). `band_source ∈
  {sigma_repro, sigma_fold}`. 보수적(넓은)이라 안전하며, 리포트 인용값(pooled)과
  점추정이 일치해 매크로/마이크로 괴리가 없다.
- **방향 일관성 게이트를 추가한다**: magnitude 가 real_gain 이라도, 후보가
  양쪽에 엔티티가 있는 fold 중 `min_consistency_frac`(기본 2/3) 미만에서만
  이기면 INCONCLUSIVE 로 내린다(real_regression 도 대칭). 한 fold 가 pooled 를
  끌어올린 경우를 거른다.
- **불변식**: 일관성 게이트는 verdict 를 **조일 뿐 절대 풀지 않는다** —
  within_noise 를 gain 으로 올리지 않고, gain 을 within_noise/inconclusive 로만
  내린다. 따라서 false PASS 를 새로 만들 수 없다. paired-SEM·매크로 점추정을
  아예 안 쓰므로 SEM 붕괴·small-n·점추정 불일치가 전부 소멸한다.
- fold 별 Δᵢ 는 양쪽 run 이 같은 fold 멤버십일 때만 짝지을 수 있다(=comparable).
  Part A 가 seed·stratify·지문을 자에 넣어 이를 강제하므로 둘이 서로를 지탱한다.
  겹치는 fold 가 <2 면 일관성 판정 불가 → 다운그레이드 없이 null 로 보고.

### 수락 기준 (2차)

- [ ] 7. `metrics.json` 이 `data_fingerprint`·`stratify` 를 기록한다. 지문은
  **전체 --data 파일 내용의 순서 민감 해시**(행별 text + 정렬된 (label,start,end)
  를 파일 순서대로). `--data-extra-train-jsonl`(train 전용)은 test gold 를 안
  바꾸므로 지문에서 **제외** — with/without extra-train 두 run 은 같은 지문이라
  비교 가능해야 한다.
- [ ] 8. `check_comparable` 이 `RULER = (lang, data_fingerprint, kfold,
  group_key, seed, stratify)` 로 판정한다. 경로 같고 **내용·순서** 다름 →
  INVALID, 경로만 rename(내용·순서 동일) → comparable, 지문 없는 옛 산출물 →
  fail-loud INVALID. `data_path` 는 자에서 제외(참고용).
- [ ] 9. 점추정 = pooled Δ(변경 없음), 밴드 = σ_repro override 또는 σ_fold
  (변경 없음). **추가**: paired-fold 방향 일관성 게이트가 magnitude gain/
  regression 을 fold 승률 <2/3 일 때 INCONCLUSIVE 로 내린다. 게이트는 조이기
  전용(불변식). compare 는 엔티티별 `fold_wins`·`fold_losses`·`fold_n`·
  `consistent` 를 보고한다.
- [ ] 10. 무회귀 + 테스트: 지문 일치→comparable/불일치→INVALID/순서 치환→INVALID,
  extra-train 제외로 같은 지문, 옛 산출물 fail-loud, **일관성 게이트가 절대
  upgrade 하지 않음**(within_noise 유지), 한 fold 가 끌어올린 pooled gain 이
  INCONCLUSIVE 로 내려감, 기존 테스트 갱신·통과.

### 2차 구현·검증

| 파일 | 변경 |
|---|---|
| `data_utils.py` | `dataset_fingerprint(rows)` — 내용의 순서 민감 sha256(행별 text + 정렬된 (label,start,end)) |
| `__main__.py` | `metrics.json` 에 `data_fingerprint`·`stratify` 기록 |
| `variance.py` | `RULER` 에 지문·seed·stratify(data_path 제외) / `fold_paired_deltas` / 방향 일관성 게이트(조이기 전용) / notes 갱신 |

**수락 기준 (2차)**

| # | 기준 | 결과 |
|---|---|---|
| 7 | `metrics.json` 이 지문·stratify 기록 | ✅ (smoke 로그에 지문 확인, 정적 검증) |
| 8 | 지문·seed·stratify 로 판정, 경로 제외 | ✅ 경로만 다름→comparable, 지문 다름→INVALID, 지문 없음→fail-loud |
| 9 | pooled Δ 점추정 유지 + 일관성 게이트(조이기 전용) | ✅ |
| 10 | 무회귀 + 테스트 | ✅ 461 통과, ruff clean |

**실검증**: 현재 `pii_all.jsonl` 지문 `2a5d0f21…`. 정답 1개만 바꿔도 지문
변화, 경로만 바꾸면 불변, 엔티티 나열 순서는 불변. 학습은 이 환경에서 모델
다운로드가 막혀 완주 못 했으나 지문이 학습 전 로그에 찍히고 metrics 기록
경로는 정적 검증(변수 정의 197·198 → summary 449).

**정의-시점 반박자가 최초 Part B 를 부쉈다** — paired-SEM 밴드는 std→0 붕괴·
small-n anticonservative·점추정 불일치로 false PASS 를 낳는다. 방향을 뒤집어
**조이기 전용 일관성 게이트**로 재설계했고, 그 결과 지적된 통계 결함이 전부
소멸했다(paired-SEM·매크로 점추정을 안 쓰므로).

## 범위 밖 (2차 이후에도)

- KO ORG gold 상태 (자를 바꾸는 별건 — 사람 소유)
- 학습 없이 분할만 검증하는 독립 진입점
- σ_repro 를 실제 측정해 캐시 (수요기반 — INCONCLUSIVE 가 채택을 막을 때만)

## 남은 위험

**그룹 키를 아예 기록하지 않으면 이 검사도 못 잡는다.** 증강 시 새 행에 뿌리
번호를 물려주지 않으면 어떤 필드도 형제를 묶지 못하고, 검사는 "형제 없음" 으로
통과한다. 재작성된 문장에는 뿌리의 흔적이 남지 않으므로 사후에 복구할 수도 없다.
데이터를 만드는 쪽이 지켜야 하는 약속이며, 검사기가 대신해줄 수 없다.
