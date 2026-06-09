# 0001. JA PROD 정의에서 서비스·온라인 운영물·기술 표준 제외

- Status: Accepted
- Date: 2026-06-08
- Deciders: groovallstar (project owner)

## Context

PROD 는 #73 에서 "gold 천장(측정 노이즈 + schema 미규정으로 인한 gold
비일관)"으로 결론났고, #84(§3.1)에서 제조 artifact·법령·전시·프로젝트
4범주를 규정했다. 그 뒤 재감사에서 **남은 비일관의 주범이 "서비스"**임이
드러났다.

원본 Stockmark `製品名` 은 상품·서비스·작품·소프트웨어를 한 라벨로
뭉뚱그렸고, canonical §1 도 "서비스"를 PROD 에 포함하고 있었다. 그러나
현재 PROD 1042 occurrence 를 LLM 다단 분류(스팟체크·반박 검증 포함)한
결과:

- 통신·SaaS·웹/온라인 매체·금융상품·멤버십·검정·강좌 등 **무형 서비스**,
- 온라인 운영형 게임(MMO),
- 기술 표준·규격·포맷·프로토콜·라이선스

가 PROD 안에 비일관하게 섞여 있었고(동 표면형이 라벨/무라벨 혼재),
이들이 회색지대의 핵심이었다. 또한 상·훈장·규격기·회사·계획 등
**명백한 오라벨**도 PROD 로 들어가 있었다.

## Decision

PROD 를 **어노테이션 대상만 positive 로 명시**하는 방식으로 재규정한다.

PROD = 유형 제품(시판 물품·기기·하드웨어·식품·약품) · 창작 작품(영화·
음악·서적·만화·게임·방송 프로그램) · 패키지 소프트웨어/OS · 형식·모델명
제조물(탈것·무기·함정·항공기·위성·기관차).

다음은 PROD 가 **아니다** (귀속처):

- 무형 서비스·온라인 운영물(웹·SaaS·MMO·통신·금융·멤버십·검정·강좌),
  기술 표준/포맷/프로토콜/라이선스 → **비-entity**
- 회사·사업체·레코드 레이블 → **ORG**
- named 계획·프로젝트·전시·레이싱 시리즈 → **EVT**
- 상·훈장·작위·규격기·생물 통칭 → **비-entity**

순 PROD support 1042 → 943 (이탈 99 = 재라벨 20[ORG 10·EVT 10] + 제거
79). 라벨링 프롬프트(`ner_prompts.py`)도 동일 positive 정의로 통일하되
"~~ 제외" 나열이 아니라 어노테이션 대상만 명시한다.

## Consequences

**측정(prodclean 10-fold, single-run):**

| 지표 | prodschema | prodclean | baseline |
|---|---:|---:|---:|
| PROD F1 | 0.8090 | 0.8168 | 0.8004 |
| overall F1 | 0.9222 | 0.9251 | 0.9256 |
| ORG F1 | 0.8913 | 0.8965 | 0.8982 |
| EVT F1 | 0.8366 | 0.8353 | 0.8593 |

- **긍정**: overall 이 baseline 수준으로 회복(0.9251), overall P 최고
  (0.9111), ORG 회복. gold 일관성·방어가능성 확보.
- **주의(비교 불가)**: PROD F1 상승은 **가장 비일관·난해했던 79 스팬을
  제거**한 데서 오는 기계적 효과 — support 정의가 바뀌어 baseline 0.8004
  와 **비교 불가**. "PROD 성능 개선"으로 보고하지 말 것. 이 변경은
  성능 레버가 아니라 **gold 일관성 보정**이다.
- **리스크 해소(#85)**: EVT −2.4pp 는 baseline `loccensus_refonly`
  (support 968) vs prodclean(987) 의 **cross-baseline** 수치였다. #84
  서비스제외의 격리 효과는 동일 계열 prodschema(0.8366)→prodclean
  (0.8353) = −0.13pp(seed42, 노이즈). multi-seed 폐기, EVT 회색지대
  규정으로 전환 (#85, ADR 0002).
- **출처 이탈**: gold 가 원본 Stockmark `製品名` 에서 멀어진 **프로젝트
  고유 재주석**이 됨(~10% PROD 스팬 변경). 의도된 선택.

**후속 작업:**

- multi-seed 검증 PASS 시 `pii_all_phonediv_prodclean.jsonl` 을 권위
  gold 로 승격(백업 `.preprodclean`).
- VI 라벨러(`src/ner/labelers/vi/`)·`augmenters/wikiann_vi` 재라벨
  프롬프트도 동일 positive PROD 정의 반영(cross-lang 일관성).
- 프롬프트 EVT/ORG 정의도 §3.1 C/D 재배치 반영 검토(named 계획·전시
  → EVT).

## Alternatives Considered

- **현상 유지(서비스 포함, Stockmark `製品名` 그대로)**: 회색지대 비일관이
  측정 천장의 원인이라 기각.
- **PROD = 상품명만 한정**: PROD 의 ~75%(작품)를 삭제 → 작품 추출 능력
  회귀, support 1042→~130, 골대 옮기기. 기각.
- **PROD → PROD(상품)+WORK(작품) 분할(11종)**: flat-10 계약 파기 +
  VI 재라벨 + 전수 재판정 비용. 기각.

## References

- `docs/manual/data/canonical-entity-schema.md` §1, §3.2, §4.1
- `src/ner/labelers/ja/ner_prompts.py` (PROD positive 정의 6 사이트)
- `docs/decisions/0001-prod-service-exclusion-mapping.json` (relabel 매핑)
- `results/classifier/ja_sweep/kfold10_phonediv_prodclean/pooled_metrics.json`
- gold(로컬·gitignore): `data/stockmark/pii_all_phonediv_prodclean.jsonl`
- 관련: 이슈 #84(§3.1 4범주), #73(PROD 천장)
