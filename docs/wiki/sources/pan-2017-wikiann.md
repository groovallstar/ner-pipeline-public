# Pan et al. (2017) — Cross-lingual Name Tagging and Linking for 282 Languages

- **저자**: Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, Heng Ji
- **연도**: 2017
- **학회**: ACL 2017 (Vancouver)
- **링크**: https://aclanthology.org/P17-1178/
- **유형**: 1차 문헌 (데이터셋 구축 방법론)

## 핵심 요지

위키피디아의 282개 언어판을 활용해 **silver-standard NER 어노테이션**을 자동 생성하는 cross-lingual 프레임워크. 영어 위키에서 추출한 엔티티 멘션과 KB 링크를 위키피디아의 언어 간 링크 구조를 통해 다른 언어로 전파한 뒤 self-training으로 정제. 결과물이 흔히 **WikiANN(또는 PAN-X) 데이터셋**으로 불린다.

## NER 부분 (본 프로젝트 관점)

- **태그 체계**: 표준 3종 — `PER`(Person), `LOC`(Location), `ORG`(Organization). 본 파이프라인은 정규화 단계에서 KLUE 스킴(`PS`/`LC`/`OG`)으로 매핑.
- **토큰화 단위**: **어절 단위(word-level, 공백 분할)**. 본 위키 `concepts/bio-tagging.md` §4.3 사례의 직접 근거.
- **품질 특성**: silver-standard라 일부 노이즈/경계 오류 존재. 어절 단위라 조사·후치사가 엔티티 span에 포함되는 구조적 한계가 있음.

## 인용하는 위키 페이지

- `concepts/bio-tagging.md` — 어절 단위 사례 및 한계
