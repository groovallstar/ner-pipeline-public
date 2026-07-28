# src/ner/validity/ — 실험 비교 유효성 게이트

K-fold 실험 결과를 **재학습 0회**로 검증한다. 기존 산출물(`fold{N}/metrics.json`·`pooled_metrics.json`)만 읽어 두 가지를 판정한다 — (1) 두 실험을 **비교해도 되는가**(같은 평가 기준으로 쟀나), (2) 개선이 **노이즈인가 실측인가**.

> 루트 `CLAUDE.md` 가 "실험을 재는 **평가 기준**"이라 부르는 것을 이 패키지 코드는 **자(RULER)** 라 부른다 — 같은 개념이다. `metrics/`(한 run 의 F1 계산)를 한 줄도 import 하지 않는 다른 관심사라 최상위 패키지로 분리했다.

## 책임 경계

- **포함**: 두 실험의 비교 가능성 판정 / 누출 검증 / 노이즈 밴드 대비 Δ 판정 / 셋을 묶은 최종 verdict
- **제외**: 한 run 의 F1 계산(→ `metrics/`) / 학습·재학습 / gold·split 생성 / **어떤 검증 조합을 통과해야 '실험 완료'인지의 선언**(→ 평가 기준을 바꾸는 하네스의 몫, 아래 "이 패키지가 하지 않는 것")
- **결합도**: 얕음 — `metrics/` import 0. 산출물 JSON 만 읽는다.

## 모듈

| 파일 | 역할 |
|---|---|
| `comparability.py` | 같은 평가 기준으로 쟀나. `RULER_FIELDS=(lang, data_fingerprint, kfold, group_key, seed, stratify)` 중 하나라도 다르면 비교 거부. 키가 아예 없으면(스키마 드리프트) 조용히 통과 않고 fail-loud |
| `leakage.py` | pooled cross-fold 누출 카운터 + 판정 근거(`leak_check_basis`). 신뢰 근거(`group`·`orig`)로 센 0 만 '검증된 0', `text`·`none`·`unknown` 의 0 은 '볼 수단이 없었음'이라 미검증 |
| `variance.py` | 노이즈 밴드 재료 — `σ_fold`(fold 간 std, 넓은 기본 프록시) · `σ_repro`(시드 반복 pooled 헤드라인 흔들림, 수요기반 정밀 밴드) · paired fold Δ · 방향 일관성(조이기 전용) |
| `gate.py` | `compare` — 세 축 통합 verdict(`PASS`/`FAIL`/`INVALID`/`INCONCLUSIVE`) |
| `_common.py` | fold/pooled `metrics.json` IO 공유 헬퍼 (순수 함수) |
| `__main__.py` | CLI 진입점 — `std`/`repro`/`compare` 서브커맨드를 파싱해 각각 `fold_std`, `repro_std`·`write_sigma_repro`, `gate.compare` 로 dispatch. 결과는 JSON 으로 stdout, `--out` 지정 시 파일에도 쓴다 |
| `__init__.py` | 공개 API re-export (`from ner.validity import compare`) |

## 핵심 개념

**평가 기준(자, `RULER_FIELDS`)** — 두 실험이 "같은 자로 쟀는지"를 정하는 6개 필드. 하나라도 다르면 비교 자체가 무효(`INVALID`)다.

- `data_fingerprint`: test gold 내용·순서의 지문. **경로가 아니라 정답 정체성**을 비교한다 — gold 를 고치면(같은 경로여도) 다른 자가 되어 옛 실험과 비교 거부, 반대로 rename 만이면(내용 동일) 같은 자.
- `seed`·`stratify`·`group_key`·`kfold`·`lang`: fold 멤버십을 결정. paired Δ(fold 별 짝짓기)가 성립하려면 이들이 같아야 하므로 자에 포함된다 → **variance 는 comparability 에 의존한다**(comparability 없이 paired Δ 를 쓰면 안 됨).

**노이즈 밴드** — Δ = pooled per-entity F1(candidate − baseline). 밴드 = `band_k × σ`(기본 `band_k=2.0`). σ 는 엔티티마다 `σ_repro`(있으면) 아니면 `σ_fold` 로 폴백하고 `band_source` 로 남긴다. σ_fold 는 σ_repro 의 더 거친·대체로 넓은 프록시라 "노이즈 게이트는 넓게 틀리는 게 안전"하다는 원리로 기본값이 된다 — σ_repro 는 `INCONCLUSIVE` 가 **채택할 결정을 막을 때만** 측정한다.

**방향 일관성(조이기 전용)** — magnitude 가 real_gain 이어도 후보가 양쪽에 엔티티가 있는 fold 중 `MIN_CONSISTENCY_FRAC`(2/3) 미만에서만 이기면, 한 fold 가 pooled 를 끌어올린 것으로 보고 within_noise(`INCONCLUSIVE`)로 내린다. **절대 올리지 않고 regression 은 안 건드린다 → false PASS 를 못 만든다.**

## verdict 판정 (`compare`)

우선순위 순서(위에서 걸리면 그것으로 확정):

| 조건 | verdict |
|---|---|
| 비교 불가(자 다름) | `INVALID` |
| 누출 미검증(신뢰 근거 0 미달) | `INVALID` |
| 누출 관측(카운터 > 0) | `FAIL` |
| 타깃 엔티티가 pooled 에 없음 | `INVALID` |
| 타깃이 real_regression | `FAIL` |
| 타깃 외 real_regression 존재(collateral) | `FAIL` |
| 타깃이 real_gain | `PASS` |
| 그 외 | `INCONCLUSIVE` |

누출은 '검증했고 0'만 통과다. 미측정(`None`)·약한 근거의 0 은 검증된 0 이 아니다 — 정직하게 opt-out 한 run 이 벌받고 거짓 0 이 통과하면 규칙이 편법을 보상하기 때문. 다만 카운터가 0 보다 크면 근거가 약해도 **본 것**이라 `FAIL`(약한 근거는 누출을 놓칠 뿐 없는 누출을 만들지 않으므로).

## CLI

```bash
# 한 run 의 per-entity fold 간 std (σ_fold)
python -m ner.validity std --run <run_dir> [--matching strict|relaxed]

# 시드 반복 CV run 들의 재현 분산(σ_repro) → 캐시 저장
python -m ner.validity repro --runs <run1> <run2> [...] \
    [--matching strict|relaxed] [--out <sigma.json>]

# candidate 를 baseline 대비 게이트
python -m ner.validity compare \
    --baseline <base_dir> --candidate <cand_dir> --target <ENTITY> \
    [--matching strict|relaxed] [--band-k 2.0] \
    [--sigma-repro <sigma.json>] [--out <verdict.json>]
```

`--matching`(기본 `strict`)은 세 서브커맨드 모두에 있다 — 한 비교 안에서는 같은
값으로 통일해야 한다. strict 로 잰 σ 를 relaxed Δ 에 밴드로 대면 자가 어긋난다.

`σ_repro` 는 (데이터×아키텍처×학습설정) setup 의 성질이라 setup 당 한 번만 재서(`repro --out`) 이후 `compare --sigma-repro` 로 재사용한다.

## 입력 계약

`--run`/`--baseline`/`--candidate` 디렉토리는 `src/ner/classifier`(`kfold_pool`)가 낸 K-fold 산출물이다:

- `fold{N}/metrics.json` — per-entity·overall F1(`per_entity_strict`/`per_entity` 등) + `RULER_FIELDS` 값(첫 fold 가 run 전체 대표: split seed 는 fold 마다 안 바뀜)
- `pooled_metrics.json` — `{strict,relaxed}.per_entity`·`.overall` + `cross_fold_group_dups`(누출 카운터) + `leak_check_basis`(근거)

누출 근거·그룹 키의 상세는 `src/ner/classifier/CLAUDE.md` §누출-free 분할 참조.

## 이 패키지가 하지 않는 것 (경계)

세 검증은 각자 독립 함수다. **어떤 부분집합을 통과해야 '실험 완료'인지의 조합·우선순위·선언은 이 패키지의 몫이 아니다** — 그것은 채점규칙(평가 기준)을 바꾸는 하네스가 정하며, 루트 `CLAUDE.md` 의 "검사 게이트"가 그 진입점이다. 특히 variance 의 paired Δ 는 comparability 가 성립할 때만 유효하므로, 하네스는 variance 를 comparability 없이 조합해선 안 된다.

## 테스트

```bash
python -m pytest tests/ner/test_validity.py -q
```

- golden 특성화: `tests/ner/golden/validity/*.json` — verdict 별 대표 케이스(`pass_clean_gain`·`fail_collateral_regression`·`invalid_group_key_mismatch`·`inconclusive_*` 등)를 byte-고정한다. 이 패키지는 `metrics/variance` 에서 승격됐으며, 이 골든이 이동 전후 verdict 불변을 잠근다.

## 주의

- `PYTHONPATH` 설정·주입 금지 — uv editable install 이 자동 등록
- import: `from ner.validity import compare` (src 접두어 없이)
- print/log/argparse help: 영문 / docstring·주석: 한국어 (`docs/specs/coding-conventions.md`)
