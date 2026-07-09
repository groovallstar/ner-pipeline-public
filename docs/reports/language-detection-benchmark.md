# 언어 감지(ja·vi·unsupported) 후보 벤치마크

- 측정일: 2026-06-26
- 대상: `server.detect` 감지기 후보 2종 — 손규칙(codepoint) vs
  fastText-LID(`lid.176.ftz`)
- gold 평가셋: FLORES-200 dev 샘플(자연어 10종) + 파생 4종, 1,398문장
- 재현: 측정에 쓴 하네스(`server.scripts.lang_detect` — gold 빌더·후보·
  혼동행렬·벤치)는 손규칙 채택이 확정된 뒤 폐기했다. 재실행 트리거가 없는
  1회성 도구이고, 결론은 `server.detect` 의 손규칙 구현과 이 리포트에
  남는다. 다시 필요하면 git 히스토리에서 되살린다 — gold 는 공개 데이터
  (FLORES-200 dev)에서 seed 42·언어당 100문장으로 결정적으로 생성됐고,
  파생 4종(한자only-ja·무부호-vi·romaji-ja·vi+ja)은 그 샘플의 결정적 함수다.
- 원시 결과: `results/lang_detect_bench.json`(gitignore).

## 요약

ja·vi 양성 감지 + 미지원 명시로 `detect.py` 를 재설계하기 위해, 가나
override 를 공유하는 두 후보를 같은 gold 셋으로 비교했다. **두 후보 모두
하드 목표를 충족**한다 — ja-kana recall 100%, pan-Latin(fr/pt/de/es/tr)
vi false-accept 0, vi-부호 recall 100%, ko/en/zh/romaji-ja → unsupported
100%. 차이는 **không-dấu vi 단 한 곳**: 손규칙은 100% unsupported(결정에
정합), fastText-LID 는 7%를 vi 로 복구(결정 위반). 손규칙이 모든 측정
지표에서 fastText 이상이고 무의존·결정적·완전 설명가능이라 **손규칙을
채택**했다.

## 조건 — gold 평가셋 설계

- **자연어 10종**: FLORES-200 dev 에서 언어당 100문장 고정 seed(42) 샘플 —
  ja/vi/ko/en/zh/fr/pt/de/es/tr. 전문 번역 실문장·병렬·CC-BY-SA.
- **파생 4종**(같은 FLORES 샘플의 결정적 함수):
  - `ja_kanji_only`(98): ja 문장의 최장 한자 run(≥2) — zh 와 스크립트 동일
    스트레스. expected=ja(**보고 전용** 한계).
  - `vi_khong_dau`(100): vi 문장의 부호 전제거(NFD→결합부호 제거+đ→d).
    expected=unsupported(수용된 한계).
  - `romaji_ja`(100): pykakasi Hepburn 로마자화 + 잔여 가나(중점 ・)
    제거. expected=unsupported.
  - `vi_ja_switch`(100): vi 문장 + ja 문장 연결(가나 혼입). expected=ja
    (가나 override).
- **누출-free·non-gameable**: 손규칙은 무학습, FLORES dev 는 규칙 적합에
  미사용. `vi_diacritics` 는 술어로 필터하지 않고 샘플 전수를 쓴다(규칙에
  유리하게 고르지 않음). 측정은 coverage 율이 아니라 gold expected 라벨
  대비 혼동행렬 — 규칙 재진술(tautology) 회피.

## 결과 — 버킷별 recall(예측==expected)

| 버킷 | expected | n | 손규칙 | fastText-LID |
|---|---|---|---|---|
| ja_kana | ja | 100 | **100%** | 100% |
| vi_diacritics | vi | 100 | **100%** | 100% |
| vi_khong_dau | unsupported | 100 | **100%** | **93%** |
| romaji_ja | unsupported | 100 | 100% | 100% |
| vi_ja_switch | ja(override) | 100 | 100% | 100% |
| ko / en / zh | unsupported | 300 | 100% | 100% |
| fr / pt / de / es / tr | unsupported | 500 | 100% | 100% |
| **vi false-accept**(fr/pt/de/es/tr) | — | 500 | **0** | **0** |
| ja_kanji_only(보고 전용) | ja | 98 | 0% | 0% |

클래스별 micro P/R(전 버킷 pooled):

| 후보 | ja P/R | vi P/R | unsupported P/R |
|---|---|---|---|
| 손규칙 | 100% / 67.1% | **100% / 100%** | 91.1% / 100% |
| fastText-LID | 100% / 67.1% | 93.5% / 100% | 91.0% / 99.3% |

> ja recall 67.1%·unsupported precision 91%는 **kanji-only-ja(98) 한계가
> ja↔unsupported 로 접혀 든 결과**다(98건 모두 ja→unsupported 오라우팅).
> 운영상 의미 있는 ja-kana recall 은 100%이고, 이 한계는 가나 없는 한자
> 전용 입력이 zh 와 스크립트가 같아 복구 불가하다는 *보고 전용* 사실이다.

## 해석 — 채택 근거와 한계

- **손규칙 채택**: 측정 지표가 모두 fastText 이상이고, không-dấu 에서만
  갈리는데 거기서 손규칙이 결정("không-dấu=수용된 한계→unsupported")에
  정확히 정합한다. fastText 의 7% vi 복구는 *결정 위반*이자 비일관(93%만
  복구)이라 예측가능성을 해친다. 손규칙은 fasttext 의존·938KB 모델·
  numpy-2 마찰 없이 동일 성능을 낸다. fastText 의 유일한 잠재 이점(라틴
  스크립트 언어 en 의 *존재* 감지)은 현 범위 제외라 미발현.
- **수용된 한계 3종**: ① không-dấu vi(부호 없으면 en 과 구분 불가) →
  unsupported. ② kanji-only ja(가나 없으면 zh 와 동일) → 복구 불가.
  ③ 부호 빈약 짧은 vi(예 "Xin chào" — grave 뿐) → unsupported. 모두
  false-accept 0 을 위한 좁은 술어의 대가이며 full 문장에선 거의
  무영향(FLORES vi 100문장 전수 감지).
- **부수 관찰**: 가타카나 중점 ・(U+30FB)은 가나 블록이라 가나 신호로
  잡힌다 — `Jon・Grant` 같은 입력은 ja 로 본다(일본어 고유 구두점이라
  양성 신호로 수용). gold romaji 버킷은 이 잔여 ・를 제거해 '가나 없는
  romaji' 불변식을 유지한다.
- **범위 밖 변별 중복**(테스트 5종엔 없음, 참고): dot-below 는 Yoruba·
  전사표기, đ 는 크로아티아·세르비아 라틴과도 겹친다 — fr/pt/de/es/tr
  하드 0 은 무관하나, 향후 그 언어들을 지원셋에 넣으면 재측정 필요.
