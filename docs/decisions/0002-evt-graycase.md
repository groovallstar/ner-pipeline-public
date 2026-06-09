# 0002. JA EVT scope 재정의 — 회색지대 규정

- Status: Accepted
- Date: 2026-06-09
- Deciders: groovallstar (project owner)

## Context

#84(ADR 0001) PROD 서비스제외 후, 남은 미진단 NER 엔티티 EVT 의 −2.4pp
(single-run) 가 실재 회귀인지 노이즈인지 미판정으로 #85 가 multi-seed
재확인으로 출발했다. 그러나 진단 결과 둘이 드러났다.

1. **−2.4pp 는 cross-baseline 이었다.** baseline `0.8593` 은
   `kfold10_phonediv_loccensus_refonly`(EVT support 968, #84 *이전* gold)
   이고 prodclean(0.8353)은 support 987 — gold lineage 단계·support 가
   달라 직접 비교 불가. #84 서비스제외의 **격리 효과**는 동일 gold
   계열인 prodschema(0.8366)→prodclean(0.8353) = **−0.13pp**(seed42,
   노이즈 바닥). multi-seed 40런은 폐기.

2. **EVT 자체가 회색지대였다.** 정본 gold EVT 987 스팬을 형태별 분해한
   결과, canonical §1(1회성 행사·전쟁·조약·대회)이 명시 규정하지 않은
   군집이 다수 — 자연재해(37)·경제위기(12)·선거(generic)·지속상태
   (`冷戦`·`時代`·`問題`) 가 라벨/무라벨·타입 혼재. PROD §3.1 처럼 EVT
   경계를 규정해 gold 비일관을 닫는다.

## Decision

EVT 를 **1회성·bounded 행사·사건**으로 positive 명문화하고, **지속 상태·
시대·추상 topic 을 제외**한다. canonical §3.3 신설.

**EVT 포함 (명문화):**

- 자연재해·대형사고 (1회성 `事件`)
- named 경제·금융·정치 위기 (1회성 `事件`)
- 주기적 복합 행사명사: 선거·투표·조사 (`総選挙`·`国勢調査` 등 — generic
  단독 예외, 제도화된 주기적 공식 행사를 지시)
- 경기·컵·토너먼트·챔피언십·`歌合戦`·`甲子園` (지속 운영리그는 ORG)
- dated bounded 운동·사건·학살 (`五四運動`·`南京大虐殺`)

**EVT 아님 (비-entity):**

- 추상 논쟁 topic (`〜問題`), 세기 시대 (`〜時代`), 다년 지속 process·
  state (`冷戦`·`宗教改革`·`産業革命`·`ホロコースト`)
- 무형 서비스·제품개발 코드네임 (`ポストカプセル郵便2001`·
  `プロジェクト・ミッドウェー`, §3.2 연장)

**gold 변동 (vs prodclean = `.preevtgray`, EVT 987):**

- EVT 987→992: 제거 14(EVT→∅) + 편입 19(ORG→EVT 7 · PROD→EVT 1 ·
  완성 11[선거·투표·재해 family])
- ORG 5310→5303 (−7, 챔피언스리그·甲子園), PROD 943→942 (−1, NHK紅白歌合戦)

## Consequences

**측정(evtgray 10-fold, seed42):**

| 지표 | prodclean | evtgray | Δ |
|---|---:|---:|---:|
| EVT F1 | 0.8353 | 0.8332 | −0.21pp |
| overall F1 | 0.9251 | 0.9244 | −0.07pp |
| ORG F1 | 0.8965 | 0.8956 | −0.09pp |
| PROD F1 | 0.8168 | 0.8149 | −0.19pp |

전 엔티티가 노이즈 바닥(±) 내 무변 — gold 천장 재확인.

- **비교 불가**: EVT/ORG/PROD support 가 동시에 바뀌어(987→992 · 5310→
  5303 · 943→942) ΔEVT 를 단독 격리할 수 없다. "EVT 성능 개선/회귀"로
  보고하지 말 것 — 성능 레버 아닌 **gold 정의·일관성 보정**.
- **출처 이탈 심화**: ADR 0001 에 이어 EVT 도 원본 Stockmark `イベント名`
  에서 멀어진 프로젝트 고유 재주석이 됨. 의도된 선택.

## Alternatives Considered

- **multi-seed(s42·43·44) 로 −2.4pp 확인**: −2.4pp 가 cross-baseline 임이
  드러나 무의미. 폐기.
- **EVT 회색지대 방치(현 gold 유지)**: 비일관이 측정 천장의 원인. 기각.
- **경기·리그 전부 EVT**: `Jリーグ`(운영 조직)까지 EVT 는 "리그=지속
  조직" 원칙 파기. 경기성 명칭만 EVT 로 한정.

## References

- `docs/manual/data/canonical-entity-schema.md` §3.3, §1
- `docs/decisions/0002-evt-graycase-mapping.json` (relabel 매핑)
- `src/ner/labelers/ja/ner_prompts.py` (EVT positive 정의)
- `results/classifier/ja_sweep/kfold10_phonediv_evtgray/pooled_metrics.json`
- gold(로컬·gitignore): `data/stockmark/pii_all_phonediv.jsonl`
  (백업 `.preevtgray` = prodclean 987)
- 관련: 이슈 #85, #84(ADR 0001), #73(PROD 천장)
