# 한국어 엔티티 라벨링 규칙 (KO Entity Labeling Rules)

**LOC/ORG/시설 경계의 정본은 `canonical-entity-schema.md` §2.5 다.** 이 문서는
그 규칙을 gold 에 적용하는 **절차**만 남긴다.

정의를 여기 두지 않는 것은, 같은 규칙이 두 곳에 있으면 한쪽만 고쳐지기 때문이다.
게다가 §2.5 는 기준 파일이라 고치려면 사람 승인과 반박자를 지나지만 이 문서는
그렇지 않아서, 정의가 여기 남아 있는 동안에는 **자를 바꾸는 데 아무 마찰이 없다.**

§2.5 로 옮긴 것 — 한 줄 원칙(자연·행정 지리 = `LOC` · 정부·행정·공공·정치 기관 =
`ORG` · 그 외 전부 비-entity) · LOC/ORG/비-entity 판정표 · 환유 정부건물명
allowlist 6 종 · 회색지대 per-instance 판정축(`청`/`서` · `원(院)` · `관(館)` ·
`센터`·`연수원` · `공사`/`공단` · `세종대로`류) · 단독 표면형(`정부`·`경찰`·`군`).

EVT 는 처음부터 이 문서 범위 밖이며 정본은 canonical §3.3·§5.3 이다.

## 적용 (dual-LLM 전수 재라벨)

원본 gold 의 모든 LOC/ORG/시설 인스턴스를 §2.5 를 프롬프트로 주입한
dual-LLM(독립 2모델, temperature 0)으로 재판정한다. **합의분만** 반영하고
불일치는 HOLD→제거(stale 유지 금지). dual-LLM 은 "이 규칙의 전수 적용기"이지
자유 판정자가 아니다. held-out 인간 감사로 재라벨 품질(precision/recall)을
보고한다. 상세·측정: `docs/issues/issue-153-ko-loc-org-facility-schema.md`.
