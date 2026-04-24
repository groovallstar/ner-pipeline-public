# 핸드오프: 베트남어 NER conflict 스팬 신뢰도 개선 (차기 세션용)

> 작성일: 2026-04-22
> 선행 이슈: #10 (머지 대기 중, PR #11)
> 목적: 이슈 #10 `recall` 정책에서 drop된 **conflict 192건**을 자동으로
>   해결해 학습 데이터 품질·양을 늘리는 방법을 차기 세션에서 구현한다.

## 1. 맥락 요약 (현 상태)

- 이슈 #10에서 WikiANN-vi 10K test를 Gemma + Qwen 두 모델로 재라벨
- `merge_confidence.py`로 span을 4 카테고리로 분류:
  - `high` (둘 다 일치): **8,996건**
  - `conflict` (offset 같고 타입 다름): **192건** ← **본 핸드오프 대상**
  - `medium_recall` (Gemma만): 2,441건
  - `medium_prec` (Qwen만): 686건
- 현재 최종 데이터셋 (`vi_wikiann_8type_recall.jsonl`)은 `recall` 정책 →
  conflict 192건을 **제외**한 11,437건
- 본 핸드오프는 **그 192건을 자동 해결**해 데이터 품질·양을 모두 늘리는 작업

## 2. 채택 방법: 1 + 2 혼합 (자동, 저비용)

### 방법 1: Wikidata 앵커 투표
conflict span 중 Wikipedia 페이지가 있는 것에 한해, Wikidata P31(instance of)이
가리키는 타입을 **tie-breaker**로 사용.

```
예: "Becamex Bình Dương" → Gemma "法人名" vs Qwen "その他の組織名"
    → Wikidata Q813571 P31=Q476028 (association football club)
    → WIKIDATA_TO_STOCKMARK 매핑 → "その他の組織名"
    → Qwen 채택 확정
```

- 예상 해결 비율: conflict 192건 중 ~95건 (anchor mapped 비율 49% 감안)
- 이미 `src/augmenters/wikiann_vi/wikidata_anchor.py`와 134 Q-ID 매핑 보유

### 방법 2: 규칙 기반 우선순위
매핑 스펙(`docs/manual/data/vietnamese-ner-8types.md` §3.3·3.4)의 접두어
규칙을 conflict 판정에 직접 적용.

```
Bệnh viện X        → 施設名 강제
Công ty X / Tập đoàn → 法人名 강제
Đảng / Bộ / Cục    → 政治的組織名 강제
FC X / CLB X       → その他の組織名 강제
Sân bay / Ga       → 施設名 강제
Chùa / Nhà thờ     → 施設名 강제
```

- 접두어 10~15개면 conflict의 상당 비율 해결
- 순수 문자열 매칭, 네트워크·LLM 호출 불필요

### 기대 효과
- 192건 중 ~130건 자동 해결 (앵커 95 + 규칙 40, 중복 제외 기준 근사)
- `high` 8996 + 추가 130 = 약 **9,126건 high-confidence silver**
- 최종 `recall` 정책 데이터셋도 11,437 + 130 = **11,567건**으로 증가

## 3. 구현 설계

### 3.1 신규 모듈: `src/augmenters/wikiann_vi/resolve_conflict.py`

주요 함수:

```python
# 방법 2: 규칙 기반
PREFIX_RULES: list[tuple[str, str]] = [
    # (정규식 또는 접두어, Stockmark 8종)
    (r'^Bệnh viện\s', '施設名'),
    (r'^Công ty\s', '法人名'),
    (r'^Tập đoàn\s', '法人名'),
    (r'^Ngân hàng\s', '法人名'),
    (r'^Hãng\s', '法人名'),
    (r'^Đài\s(truyền hình|phát thanh)', '法人名'),
    (r'^Đảng\s', '政治的組織名'),
    (r'^Bộ\s', '政治的組織名'),
    (r'^Cục\s', '政治的組織名'),
    (r'^Sở\s', '政治的組織名'),
    (r'^Ủy ban\s', '政治的組織名'),
    (r'^Tòa án\s', '政治的組織名'),
    (r'^Quốc hội', '政治的組織名'),
    (r'^Quân đội\s', '政治的組織名'),
    (r'^Liên Hợp Quốc', '政治的組織名'),
    (r'\sFC$|^FC\s', 'その他の組織名'),
    (r'\sCLB\s|^CLB\s', 'その他の組織名'),
    (r'^Trường (Đại học|đại học)\s', 'その他の組織名'),
    (r'^Đại học\s', 'その他の組織名'),
    (r'^Sân bay\s', '施設名'),
    (r'^Ga\s', '施設名'),
    (r'^Cảng\s', '施設名'),
    (r'^Bảo tàng\s', '施設名'),
    (r'^Thư viện\s', '施設名'),
    (r'^Chùa\s', '施設名'),
    (r'^Nhà thờ\s', '施設名'),
    (r'^Đền\s', '施設名'),
    (r'^Lăng\s', '施設名'),
    (r'^Trường (Tiểu học|THCS|THPT)\s', '施設名'),  # 초중고는 施設名
]

def resolve_by_rule(surface: str) -> Optional[str]:
    """표면형에 규칙 적용, 결정된 타입 또는 None."""
    for pattern, tag in PREFIX_RULES:
        if re.search(pattern, surface):
            return tag
    return None

# 방법 1: Wikidata 앵커
def resolve_by_anchor(
    surface: str,
    qid_cache: dict,
    p31_cache: dict,
) -> Optional[str]:
    """캐시된 Wikidata P31로부터 8종 타입 결정 또는 None."""
    qid = qid_cache.get(surface)
    if not qid:
        return None
    p31 = p31_cache.get(qid) or []
    return anchor_type(p31)  # wikidata_anchor 모듈의 함수 재사용

# 혼합 해결
def resolve_conflict(
    span: dict,
    qid_cache: dict,
    p31_cache: dict,
    *,
    rule_first: bool = True,  # True면 규칙 우선, False면 앵커 우선
) -> Optional[str]:
    """conflict span 하나를 받아 확정 타입 or None 반환.

    rule_first=True 시 규칙 매칭을 먼저 시도, 실패하면 앵커.
    """
    surface = span['text']
    if rule_first:
        tag = resolve_by_rule(surface)
        if tag:
            return tag
        return resolve_by_anchor(surface, qid_cache, p31_cache)
    tag = resolve_by_anchor(surface, qid_cache, p31_cache)
    if tag:
        return tag
    return resolve_by_rule(surface)
```

### 3.2 merge_confidence.py 수정

`categorize_spans` 또는 새로운 `categorize_spans_with_resolver`에서 conflict
분류 시 resolver를 호출해 **해결 성공하면 high로 승격**하고 `source`를
`"conflict_resolved_by_rule"` 또는 `"conflict_resolved_by_anchor"`로 기록:

```python
{
  "text": "...",
  "type": "...",           # resolver 확정 타입
  "confidence": "high",     # 승격
  "source": "conflict_resolved_by_anchor",
  "gemma_type": "...",      # 원래 Gemma 타입
  "qwen_type": "...",       # 원래 Qwen 타입
  "resolver_reason": "Q476028→その他の組織名"  # 디버깅용
}
```

resolver 실패 시 기존 `conflict` 그대로 유지 (recall 정책에서 여전히 drop).

### 3.3 CLI 옵션

```bash
python -m augmenters.wikiann_vi.merge_confidence \
    --gemma ...gemma_full.jsonl \
    --qwen ...qwen_full.jsonl \
    --policy recall \
    --resolve-conflicts rule,anchor \
    --wikidata-cache data/wikiann_vi_relabel/wikidata_cache.json \
    --output vi_wikiann_8type_recall_resolved.jsonl
```

`--resolve-conflicts` 옵션:
- `none`: 기존 동작 (해결 없음)
- `rule`: 방법 2만
- `anchor`: 방법 1만
- `rule,anchor`: 규칙 먼저, 실패 시 앵커
- `anchor,rule`: 앵커 먼저, 실패 시 규칙

## 4. 선행 준비 (다음 세션 시작 시)

1. **Gemma/Qwen 10K JSONL 재생성** — 현재 디렉토리 정리 과정에서 `gemma_8type_full.jsonl`·`qwen_8type_full.jsonl` 삭제됨. 핸드오프 §5 명령어로 재생성 (~2h 25m 총)
2. **Wikidata 캐시 재생성** — `wikidata_cache.json`도 삭제. 10K 앵커 재실행 시 자동 재생성 (30~60분 네트워크)

## 5. 재현 명령

```bash
# 0) 브랜치 준비 (이슈 #10 머지 후 develop에서 분기)
git switch develop
git pull origin develop
git switch -c feat/issue-{NN}-vi-conflict-resolve

# 1) Gemma 10K 재라벨 (25분)
python -m augmenters.wikiann_vi \
    --max-samples 10000 --concurrency 16 \
    --base-url http://localhost:8081/v1 \
    --model cyankiwi/gemma-4-31B-it-AWQ-8bit \
    --output data/wikiann_vi_relabel/gemma_8type_full.jsonl

# 2) Qwen 10K 재라벨 (2시간) - 백그라운드 권장
python -m augmenters.wikiann_vi \
    --max-samples 10000 --concurrency 16 \
    --base-url http://localhost:8082/v1 \
    --model cyankiwi/Qwen3.5-27B-AWQ-4bit \
    --output data/wikiann_vi_relabel/qwen_8type_full.jsonl

# 3) Wikidata 앵커 (캐시 생성 포함, 30~60분)
python -m augmenters.wikiann_vi.wikidata_anchor \
    --input data/wikiann_vi_relabel/gemma_8type_full.jsonl \
    --cache data/wikiann_vi_relabel/wikidata_cache.json \
    --json-out data/wikiann_vi_relabel/anchor_gemma_full.json

# 4) 신규: resolve_conflict 모듈 작성 + 단위 테스트
#    (본 핸드오프 §3 참고)

# 5) 병합 + conflict 해결 (신규 옵션 사용)
python -m augmenters.wikiann_vi.merge_confidence \
    --gemma data/wikiann_vi_relabel/gemma_8type_full.jsonl \
    --qwen data/wikiann_vi_relabel/qwen_8type_full.jsonl \
    --policy recall \
    --resolve-conflicts rule,anchor \
    --wikidata-cache data/wikiann_vi_relabel/wikidata_cache.json \
    --output data/wikiann_vi_relabel/vi_wikiann_8type_recall_resolved.jsonl

# 6) 검증: 해결 건수 리포트
#    `source` 분포에서 conflict_resolved_by_{rule,anchor} 건수 확인
```

## 6. 완료 조건

- [ ] `resolve_conflict.py` + 단위 테스트 작성
- [ ] `merge_confidence.py`에 resolver 통합 + `--resolve-conflicts` CLI 옵션
- [ ] 규칙 PREFIX_RULES 20~30개 수준으로 확장 (스펙 §3.3·3.4 참고)
- [ ] 10K 재실행해 conflict 192 중 실제 해결 건수 측정
- [ ] 해결 후 merged 파일의 span 분포 리포트 (high·medium·remaining conflict)
- [ ] 리포트 신규 섹션 또는 신규 리포트(`vietnamese-ner-conflict-resolution-2026-XX.md`) 작성
- [ ] 전체 pytest 회귀 + ruff clean
- [ ] 별도 PR (closes #{새 이슈 번호})

## 7. 주의사항

- **규칙과 앵커의 충돌 시 우선순위**: 기본 `rule_first=True` 권장. 규칙은
  언어 특화 관용(Stockmark의 일본어 시설 분류 기준을 베트남어에 맞게 보정)을
  담고 있고, 앵커는 범용 Wikidata 타입이라 스펙과 어긋날 수 있음
- **anchor_type 함수의 한계**: Wikipedia 리다이렉트로 타입이 왜곡되는 경우
  존재(예: 노래 페이지가 가수 페이지로 리다이렉트). conflict 케이스에서는
  리다이렉트가 오히려 유용할 수도, 노이즈일 수도 있음 — 샘플 수작업 검토
  권장
- **rule_first 적용 후 PREFIX_RULES 확장 시 주의**: 규칙이 늘어날수록 LLM의
  판단을 덮어쓰는 비율이 늘어남. 도입 전후 **샘플 30건 수작업 판정으로
  정확도 측정** 필요
- **conflict 해결을 한 타입의 '편애'로 만들지 않도록**: 원래 `conflict`는
  애매 케이스이므로, resolver가 특정 타입(예: 政治的組織名)으로 지속적으로
  밀리면 과적합 위험. 해결 후 타입 분포를 원본 Gemma 분포와 비교해 큰
  왜곡이 없는지 확인

## 8. 참고 자료

- 이슈 #10 구현 결과: `docs/issues/issue-10-vi-ner-8type-relabel.md`
- 8종 매핑 스펙 (규칙 근거): `docs/manual/data/vietnamese-ner-8types.md`
- 이슈 #10 리포트: `docs/reports/vietnamese-ner-schema-expansion-2026-04.md`
- 기존 모듈:
  - `src/augmenters/wikiann_vi/merge_confidence.py` (분류·병합·정책 필터)
  - `src/augmenters/wikiann_vi/wikidata_anchor.py` (앵커 API·134 Q-ID 매핑)
  - `src/augmenters/wikiann_vi/prompts.py` (8종 프롬프트)
- conflict 샘플 분석 (이슈 #10 리포트 §5.4): Becamex Bình Dương, Oren Lavie,
  Thư viện Quốc gia Pháp 등 15건
