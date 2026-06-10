# 이슈 #85 — JA EVT 회색지대 canonical 규정 (scope 재정의)

- area: classifier
- 브랜치: `feat/issue-85-ja-evt-graycase`
- 관련: #84, #73(PROD 천장)

## 1. 배경 — re-scope

원래 #85 는 #84 의 EVT −2.4pp(single-run) 를 multi-seed(s42·43·44) 로
재확인하려 했다. 진단 결과 둘이 드러나 scope 를 재정의했다.

### 1.1 −2.4pp 는 cross-baseline 이었다

| dir | gold | EVT F1 | EVT support |
|---|---|---:|---:|
| `loccensus_refonly` (#84 前 baseline) | PROD 1043 | 0.8593 | 968 |
| `prodschema` (#84 서비스제외 前) | PROD 1042 | 0.8366 | 977 |
| `prodclean` (#84 後) | PROD 943 | 0.8353 | 987 |

−2.4pp = 0.8593(loccensus) → 0.8353(prodclean) 인데 support·lineage 단계가
달라 cross-baseline. #84 서비스제외의 **격리 효과**는 동일 계열
prodschema→prodclean = **−0.13pp**(seed42, 노이즈 바닥). → multi-seed
40런 폐기.

### 1.2 EVT 자체가 회색지대였다

prodclean EVT 987 스팬을 형태별 분해 → canonical §1(1회성 행사·전쟁·
조약·대회) 미규정 군집 다수: 자연재해(37)·경제위기(12)·generic 선거·
지속상태(`冷戦`·`時代`·`問題`) 가 라벨/무라벨·타입 혼재. PROD §3.1 처럼
EVT 경계를 규정해 gold 비일관을 닫는다.

## 2. 규정 (사용자 비준) — canonical §3.3

**EVT 포함 명문화**: 자연재해·대형사고 / named 경제·금융·정치 위기 /
주기적 복합 행사명사(선거·투표·`国勢調査` — generic 단독 예외) / 경기·
컵·토너먼트·`歌合戦`·`甲子園`(지속 운영리그는 ORG) / dated bounded
운동·사건·학살(`五四運動`·`南京大虐殺`).

**EVT 제외(비-entity)**: 추상 논쟁 topic(`〜問題`) / 세기 시대
(`大航海時代`) / 다년 지속 process·state(`冷戦`·`宗教改革`·`産業革命`·
`ホロコースト`) / 무형 서비스·제품개발 코드네임(`ポストカプセル郵便2001`·
`プロジェクト・ミッドウェー`).

## 3. gold 변동 (vs prodclean = `.preevtgray`, EVT 987)

- EVT 987→**992**: 제거 14(EVT→∅) + 편입 19(ORG→EVT 7 · PROD→EVT 1 ·
  완성 11[선거·투표·재해 family])
- ORG 5310→**5303**(−7, 챔피언스리그·甲子園), PROD 943→**942**(−1,
  NHK紅白歌合戦)

## 4. 산출물 (체크리스트)

- [x] canonical §3.3 + 변경이력 (`docs/manual/data/canonical-entity-schema.md`)
- [x] `ner_prompts.py` EVT positive 갱신(3 템플릿 + `チャンピオンズリーグ`
  예시 ORG→EVT)
- [x] gold relabel (백업 `.preevtgray`, offset/overlap 무결 검증)
- [x] evtgray 10-fold 재측정 (EVT/ORG/PROD 동시 변동 → ΔEVT 격리 불가)
- [x] per-entity 진단 리포트 EVT § + #84 −2.4pp cross-baseline 규명
- [x] refuter PASS → PR closes #85

## 5. 측정 결과 (evtgray 10-fold, seed42)

| 지표 | prodclean | evtgray | Δ |
|---|---:|---:|---:|
| EVT F1 | 0.8353 | 0.8332 | −0.21pp |
| overall F1 | 0.9251 | 0.9244 | −0.07pp |
| ORG F1 | 0.8965 | 0.8956 | −0.09pp |
| PROD F1 | 0.8168 | 0.8149 | −0.19pp |

전 엔티티 노이즈 바닥(±) 내 무변 — gold 천장 재확인. EVT/ORG/PROD
support 동시 변동(987→992·5310→5303·943→942)으로 ΔEVT 단독 격리 불가나,
어느 엔티티도 노이즈를 넘지 않아 **성능 레버 아닌 gold 정의·일관성
보정**임이 확인됨. `results/.../kfold10_phonediv_evtgray/pooled_metrics.json`.

## 6. 검증 (refuter)

격리 컨텍스트 Sonnet 반박자(code-reviewer) 판정 **PASS**
(diff_hash `850351437220`, `.omc/state/refuter/850351437220.json`). 3축:

- **코드**: `ner_prompts.py` EVT 3 템플릿 + `チャンピオンズリーグ` 예시
  ORG→EVT, import·ruff 통과.
- **측정 무결성**: evtgray/prodclean/loccensus/prodschema
  `pooled_metrics.json` ↔ 문서 숫자 일치, −2.4pp cross-baseline·격리
  −0.13pp 정합.
- **gold·테스트**: gold count 992/5303/942 일치, 테스트 약화 없음.
