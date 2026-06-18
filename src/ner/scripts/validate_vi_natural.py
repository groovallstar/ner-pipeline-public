"""자연 PII 주입 코퍼스 품질 게이트 (결정론 검사 + 보존율).

LLM verify 없이 자연 주입본을 검증한다:
  1. suffix 마커 잔존율(0% 기대)
  2. span offset round-trip 정합(text[s:e]==span.text, 100% 기대)
  3. 레코드 드롭율(주입 PII 값 verbatim 실패 시 레코드 통째 드롭)
  4. 원문 NER 보존율 — source 원문 NER (label,text) 가 주입본에 재배치됐는지.
     드롭된 레코드의 원문 NER 은 전손으로 계산.
  5. 무라벨 email/phone 표면 잔존 스캔(학습 모순 후보)

source 와 injected 는 **유니크 `id`** 로 조인한다(원본 id 는 중복이 많아
`build_vi_natural_source` 가 행 인덱스로 유니크 id 를 부여). injected 가
source 앞 N 행 부분집합(--n-input)일 수 있으므로 source[:N] 을 시도 모집단
으로 잡고, 그 안에서 id 조인한다. 드롭된 행(주입 PII verbatim 실패)의 원문
NER 은 보존율 분모에 전손으로 포함한다.
"""
import argparse
import json
import re
from collections import Counter, defaultdict

NER_TYPES = ('PER', 'LOC', 'ORG', 'PROD', 'EVT')
MARKERS = (' Liên hệ:', ' SĐT:', ' Địa chỉ:', ' Ngày:', ' CCCD:',
           ' Email:', ' Thẻ:')
_EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_PHONE = re.compile(r'(?<!\d)(?:0|\+84)\d[\d .-]{7,12}\d(?!\d)')


def _load(path: str) -> list[dict]:
    return [json.loads(ln) for ln in open(path, encoding='utf-8')]


def validate(source: list[dict], injected: list[dict],
             n_input: int) -> dict:
    """게이트 지표 dict 산출. source[:n_input] 을 시도 모집단으로 평가."""
    inj_by_id = {r.get('id'): r for r in injected}
    if len(inj_by_id) != len(injected):
        raise ValueError('injected ids are not unique — join unreliable')
    src_eval = source[:n_input]

    marker_rows = sum(
        1 for r in injected if any(m in r['text'] for m in MARKERS))
    offset_bad = 0
    unlabeled_email = unlabeled_phone = 0
    for r in injected:
        t = r['text']
        spans = sorted(r['entities'], key=lambda e: e['start_char'])
        for e in spans:
            if t[e['start_char']:e['end_char']] != e['text']:
                offset_bad += 1
        covered = [(e['start_char'], e['end_char']) for e in spans]

        def _is_covered(s, e, cov=covered):
            return any(s < ce and cs < e for cs, ce in cov)
        for m in _EMAIL.finditer(t):
            if not _is_covered(m.start(), m.end()):
                unlabeled_email += 1
        for m in _PHONE.finditer(t):
            if not _is_covered(m.start(), m.end()):
                unlabeled_phone += 1

    # 보존율: source[:N] 의 원문 NER (label,text) vs 조인된 injected.
    # 드롭된 행은 out_set 이 비어 전손 처리된다.
    ret = defaultdict(lambda: [0, 0])  # label -> [retained, total]
    dropped_rows = 0
    for sr in src_eval:
        ir = inj_by_id.get(sr.get('id'))
        if ir is None:
            dropped_rows += 1
            out_set = Counter()
        else:
            out_set = Counter(
                (e['label'], e['text']) for e in ir['entities'])
        for e in sr['entities']:
            key = (e['label'], e['text'])
            ret[e['label']][1] += 1
            if out_set.get(key, 0) > 0:
                ret[e['label']][0] += 1
                out_set[key] -= 1

    retention = {
        lbl: {'retained': r, 'total': tot,
              'rate': round(r / tot, 4) if tot else None}
        for lbl, (r, tot) in sorted(ret.items())
    }
    tot_r = sum(r for r, _ in ret.values())
    tot_t = sum(t for _, t in ret.values())

    return {
        'input_rows': n_input,
        'injected_rows': len(injected),
        'dropped_rows': dropped_rows,
        'drop_rate': round(dropped_rows / n_input, 4) if n_input else None,
        'marker_rows': marker_rows,
        'offset_mismatch': offset_bad,
        'unlabeled_email': unlabeled_email,
        'unlabeled_phone': unlabeled_phone,
        'retention_overall': round(tot_r / tot_t, 4) if tot_t else None,
        'retention_by_label': retention,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--injected', required=True)
    p.add_argument('--n-input', type=int, default=None,
                   help='input rows fed to injector (default: all source)')
    p.add_argument('--out', default=None, help='write report JSON')
    args = p.parse_args()

    source = _load(args.source)
    injected = _load(args.injected)
    n_input = args.n_input if args.n_input is not None else len(source)
    rep = validate(source, injected, n_input)

    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f'-> {args.out}')


if __name__ == '__main__':
    main()
