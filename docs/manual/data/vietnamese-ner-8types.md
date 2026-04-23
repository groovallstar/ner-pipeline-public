# 베트남어 NER 8종 매핑 스펙 (Stockmark 스키마 정렬)

> 기준 시점: 2026-04-21
> 대상 이슈: #10 (VI WikiANN → canonical 8종 스키마 확장)
> 선행 스펙: `docs/manual/data/japanese-ner.md`,
>   `docs/manual/data/vietnamese-ner.md`

## 1. 목적

WikiANN-vi(3종: PER/LOC/ORG)를 일본어 Stockmark(8종)와 동일한 라벨 공간으로
확장해 단일 멀티링구얼 NER 시스템에서 공정 비교를 가능케 한다. 본 문서는
**8종 타입 정의·WikiANN 3종 → 8종 분할 규칙·모호 사례 처리**를 확정한다.

재어노테이션·합성 보충·검증 전략은 이슈 #10 플랜 문서
(`docs/issues/issue-10-vi-ner-8type-relabel.md`)를 참조.

## 2. 8종 엔티티 정의

라벨 표기는 **canonical 영문 축약**(`docs/manual/data/canonical-entity-schema.md`)을
그대로 사용한다. 프롬프트·UI에서만 베트남어 설명을 부가하며, 데이터셋 JSONL과
평가 로직의 `type` 필드는 canonical 문자열을 그대로 둔다. 이는 다국어
데이터셋 간 label space 일관성과 리포트 가독성 확보를 위한 것이다.

| 태그 (canonical) | 의미 | 베트남어 설명 | 예시(VI) |
|---|---|---|---|
| `PER` | 인물 | Tên người (풀네임·성·이름·별명·예명) | Nguyễn Xuân Phúc, Hồ Chí Minh, Bác Hồ |
| `CORP` | 법인 (기업·철도·방송사) | Công ty, tập đoàn, hãng hàng không, đài truyền hình, công ty đường sắt | Samsung, Vingroup, Vietnam Airlines, VTV, Đường sắt Việt Nam |
| `LOC` | 지명 (국가·도시·자연지명) | Quốc gia, thành phố, tỉnh, huyện, sông, núi, biển, đảo — **자연지명·행정지명만** | Việt Nam, Hà Nội, Đà Nẵng, Sông Hồng, Vịnh Hạ Long |
| `FAC` | 시설 (건물·역·공항·점포·병원·학교시설·사원·관광명소 등) | Tòa nhà, ga, sân bay, cửa hàng, bệnh viện, chùa, công trình du lịch | Chùa Một Cột, Sân bay Nội Bài, Bệnh viện Bạch Mai, Ga Hà Nội |
| `PROD` | 제품·서비스·소프트웨어·작품명 | Sản phẩm, dịch vụ, phần mềm, tác phẩm | iPhone 15, Honda Wave, VinFast VF8, Windows |
| `EVT` | 행사·대회·사건·전쟁·조약 | Sự kiện, giải đấu, chiến tranh, hiệp ước | Chiến tranh Việt Nam, SEA Games, Hiệp định Paris |
| `POL` | 정당·정부기관·국제기관·군대·재판소·의회 | Đảng phái, bộ/cơ quan chính phủ, tổ chức quốc tế, quân đội, tòa án, quốc hội | Đảng Cộng sản Việt Nam, Bộ Giáo dục, Quốc hội, Liên Hợp Quốc, Quân đội Nhân dân Việt Nam |
| `ORG` | 스포츠리그·스포츠팀·대학·단체 등 기타 조직 | Giải đấu, đội thể thao, trường đại học, tổ chức xã hội | V.League, Hà Nội FC, Đại học Quốc gia Hà Nội, Hội Chữ thập đỏ |

## 3. WikiANN 3종 → 8종 매핑 규칙

### 3.1 한눈에 보기

```
WikiANN PER → PER (1:1)
WikiANN LOC → LOC  OR  FAC
WikiANN ORG → CORP  OR  POL  OR  ORG  OR  FAC
(WikiANN 없음) → PROD · EVT (경로 B 또는 합성 보충)
```

### 3.2 PER → PER

직함·호칭(Ông/Bà/Chủ tịch/Thủ tướng/Tướng/GS 등)을 **제외**한 고유명 부분만
남긴다. 예명·별명(Bác Hồ 등)은 그대로 PER.

### 3.3 LOC → LOC vs FAC

**LOC**이 되는 것: 국가·행정구역·자연지명·광역 지리 개념.
- Việt Nam, Hà Nội, Đà Nẵng, Sông Mekong, Vịnh Hạ Long, Núi Phan Xi Păng

**FAC**이 되는 것: 개별 건축물·교통시설·종교시설·공공시설.
- Chùa Một Cột, Sân bay Nội Bài, Ga Hà Nội, Bảo tàng Dân tộc học

**경계 규칙**:
1. "Chùa/Nhà thờ/Đền/Miếu/Lăng" 접두어 → FAC
2. "Sân bay/Ga/Cảng" 접두어 → FAC
3. "Bảo tàng/Thư viện/Nhà hát/Công viên" 접두어 → FAC
4. "Sông/Núi/Biển/Đảo/Vịnh/Hồ" + 자연지명 → LOC
5. 행정 단위(Thành phố/Tỉnh/Huyện/Xã/Quận/Phường) + 고유명 → LOC
   (행정 지명은 전체를 단일 엔티티로: `Thành phố Hồ Chí Minh` = LOC 1개)
6. 관광 명소가 자연물(Vịnh Hạ Long) → LOC, 건축물(Chùa Một Cột) → FAC

### 3.4 ORG → CORP · POL · ORG · FAC

**CORP** (기업·영리법인):
- Công ty, Tập đoàn, Ngân hàng, Hãng, Xí nghiệp, 공공법인(Vietnam Airlines,
  EVN 등).
- 예: Samsung, Vingroup, Vinamilk, Vietnam Airlines, VTV (방송국)

**POL** (정당·정부·군·사법·입법·국제기관):
- Đảng, Bộ, Chính phủ, Ủy ban, Sở, Cục, Quốc hội, Tòa án, Quân đội,
  Công an, Liên Hợp Quốc, ASEAN
- 예: Đảng Cộng sản Việt Nam, Bộ Giáo dục và Đào tạo, Quốc hội,
  Liên Hợp Quốc, Quân đội Nhân dân Việt Nam

**ORG** (대학·스포츠·단체·리그):
- Đại học/Trường, 스포츠 팀·리그, 재단, NGO, 협회
- 예: Đại học Quốc gia Hà Nội, V.League, Hà Nội FC, Hội Chữ thập đỏ

**FAC으로 이동** (WikiANN이 ORG로 라벨했지만 Stockmark 기준은 시설):
- Bệnh viện (병원), 점포(cửa hàng) — Stockmark가 명시적으로 FAC로 분류.
- 예: Bệnh viện Bạch Mai (WikiANN ORG → Stockmark FAC)
- 주의: 종합병원을 운영하는 **법인**("Công ty TNHH Bệnh viện ...")이 나오면
  CORP, 병원 자체("Bệnh viện Bạch Mai")는 FAC.

**경계 규칙**:
1. "Đảng"/"Mặt trận"으로 시작 → POL
2. "Bộ"/"Cục"/"Sở"/"Ủy ban"/"Tòa án"/"Quốc hội" → POL
3. "Quân đội"/"Hải quân"/"Sư đoàn"/"Lữ đoàn"/"Công an" → POL
4. "Đại học"/"Trường Đại học"/"Học viện"(대학만) → ORG.
   단 초·중·고등학교(Trường THCS/THPT/Tiểu học)는 **FAC**
   (일본어 Stockmark가 학교·캠퍼스는 FAC로 처리)
5. "FC"/"Câu lạc bộ bóng đá"/리그명(V.League/J1 등) → ORG
6. "Bệnh viện" (병원 단일체) → FAC
7. 영리 기업 접두(Công ty/Tập đoàn/Ngân hàng/Hãng) → CORP
8. 방송사(Đài truyền hình/Đài phát thanh/VTV/VTC) → CORP
9. 국제기구(Liên Hợp Quốc/WTO/ASEAN/UNICEF) → POL

### 3.5 WikiANN에 존재하지 않는 타입: PROD · EVT

WikiANN-vi 원본에는 이 두 타입의 span이 **라벨되어 있지 않다**. 획득 경로:

1. **경로 B (재어노테이션, 기본)**: 8종 프롬프트로 vLLM이 원문을 재스캔해
   미라벨 엔티티를 새로 추출. 모델: `cyankiwi/gemma-4-31B-it-AWQ-8bit`
   (endpoint `http://localhost:8081/v1`, `max_model_len=8192`).
2. **합성 보충 (조건부)**: 경로 B 후 集計에서 빈도 부족 판정 시
   `augmenters/pii` 구조를 재사용해 자연 문맥으로 주입. 이슈 #10 5단계에서
   사용자 승인 후 결정.

경계 규칙:
- **PROD**: 상표명·모델명·소프트웨어명. "iPhone 15", "Honda Wave",
  "VinFast VF8". 회사명 자체(Samsung)는 CORP, 회사의 구체 제품(Galaxy S24)은
  PROD.
- **EVT**: 전쟁·조약·스포츠 대회·공식 행사. "Chiến tranh Việt Nam",
  "SEA Games", "Hiệp định Paris", "Đại hội Đảng". 일상 행사는 제외.

## 4. 모호 사례 10건

WikiANN의 라벨을 기준으로, canonical 8종에서의 최종 타입을 어떻게 결정할지
합의한 사례 표.

| # | 표면형 | WikiANN | canonical 8종 (본 스펙) | 판단 근거 |
|---|---|---|---|---|
| 1 | `Chùa Một Cột` | LOC | `FAC` | 개별 종교 건축물. Stockmark는 사원·교회를 FAC로 분류 |
| 2 | `Vịnh Hạ Long` | LOC | `LOC` | 자연지명(만·해안). 관광명소여도 자연물은 LOC |
| 3 | `Sân bay Nội Bài` | LOC | `FAC` | 공항. Stockmark는 공항을 명시적으로 FAC |
| 4 | `Bệnh viện Bạch Mai` | ORG | `FAC` | 병원 단일체. Stockmark는 "○○病院"을 FAC (법인 운영체와 별도) |
| 5 | `Đại học Quốc gia Hà Nội` | ORG | `ORG` | 대학은 Stockmark 기준 ORG (학교 건물이 아니라 조직체) |
| 6 | `V.League` | ORG | `ORG` | 스포츠 리그. Stockmark는 스포츠 리그를 ORG |
| 7 | `Hà Nội FC` | ORG | `ORG` | 스포츠 팀. Stockmark는 스포츠 팀을 ORG |
| 8 | `Đảng Cộng sản Việt Nam` | ORG | `POL` | 정당. Stockmark는 정당을 POL |
| 9 | `Bộ Giáo dục và Đào tạo` | ORG | `POL` | 정부 부처. Stockmark는 정부기관을 POL |
| 10 | `Vietnam Airlines` | ORG | `CORP` | 공기업도 영리 법인이면 CORP. 단 **국영 운송청**(Cục Đường sắt)은 POL |

추가 단골 모호 사례(참고):
- `Samsung Galaxy S24`: "Samsung" 부분이 법인명·"Galaxy S24"가 PROD으로
  분리. WikiANN은 일반적으로 "Samsung"만 ORG로 라벨. 재어노테이션에서
  제품명 부분이 신규 추출될 수 있다.
- `Thành phố Hồ Chí Minh`: 행정 지명 전체를 단일 `LOC`. "Hồ Chí Minh" 하나만
  PER으로 분리하지 않는다. 단독 "Hồ Chí Minh"(인물)은 PER.

## 5. Silver 한계 및 검증 전략

### 5.1 이중 silver 인식

- WikiANN 원본: Wikipedia 인터링크 기반 자동 silver
- 본 스펙의 8종 라벨: WikiANN silver를 LLM이 재분류 → 추가 silver
- **결과**: Stockmark(gold) vs 본 데이터셋(이중 silver). 절대 F1 비교는
  포기하고 상대 순위·카파(kappa) 지표로만 해석한다.

### 5.2 3중 검증 레이어 (이슈 #10 4단계 구현)

1. **Cross-model agreement**: 최소 2개 LLM(예: gemma-4-31B / Qwen3.5-27B)으로
   독립 재분류 → Cohen's kappa 측정. 불일치 샘플은 별도 버킷.
2. **Wikipedia 인터링크 앵커**: 베트남어 Wikipedia 엔티티 → 일본어
   Wikipedia 링크 → 일본어 페이지의 Category/Infobox로 타입 역추정
   (Wikidata P31/P279 포함). 커버리지는 제한적이지만 독립 gold 신호.
3. **PER/LOC/ORG 3종 공정 비교**: 확장 데이터셋에서 3종만 subset해서
   Stockmark 3종(PER/LOC/CORP+POL+ORG 합계)과
   비교 행을 만든다. 8종 절대값 비교는 하지 않는다.

### 5.3 명시해야 할 리포트 문구 (권장 템플릿)

> 본 데이터셋은 WikiANN(silver) 위에 LLM 재분류(silver)를 얹은 이중 silver
> 파생물이다. 일본어 Stockmark(gold)와의 F1 절대값 비교는 수행하지 않으며,
> 모델 간 상대 순위와 cross-model kappa만을 해석 대상으로 한다.

## 6. 라벨 표기·인코딩 규칙

- JSONL의 `type` 필드: canonical 영문 축약
  (`PER`, `CORP`, `LOC`, `FAC`, `PROD`, `EVT`, `POL`, `ORG`)
- UTF-8, NFC 정규화 가정(베트남어 성조 결합문자 손실 방지).
- `text` 필드는 원문을 **무손실 보존**. span 경계는 문자 오프셋
  (JA loader / VI loader 동일 스키마).

## 7. 미결 사항

- PROD·EVT 실제 빈도(재어노테이션 집계 후 결정).
- 경로 B 후 합성 보충 도입 여부 — 이슈 #10 5단계에서 사용자 승인 후 확정.
- "Sân vận động"(경기장), "Khách sạn"(호텔) 같은 경계 케이스 추가 규칙 —
  재어노테이션 실측에서 빈도 확인 후 규칙 3.3·3.4에 추가할지 판단.

## 8. 변경 이력

- 2026-04-21 초안. 이슈 #10 2단계 산출물.
- 2026-04-22 (이슈 #13) 라벨 표기를 canonical 영문 축약(PER/CORP/LOC/FAC/
  PROD/EVT/POL/ORG)으로 통일. 본문·표·예시 일괄 치환.
