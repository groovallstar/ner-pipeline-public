"""신뢰도 임계값 운영점: scored decode·apply·fit·save/load 단위 테스트."""
import json

from ner.classifier.confidence_threshold import (
    apply_thresholds,
    fit_thresholds,
    load_thresholds,
    save_thresholds,
)
from ner.classifier.data_utils import build_label_maps, decode_bio_to_spans
from ner.metrics.span_metrics import compute_offset_span_f1


def test_decode_confs_attaches_conf_mean():
    """confs 입력 시 span 에 score = 토큰 신뢰도 평균(conf_mean)."""
    label2id, id2label = build_label_maps()
    label_ids = [label2id['B-PER'], label2id['I-PER'], label2id['B-ORG']]
    offsets = [(0, 3), (3, 6), (7, 10)]
    confs = [0.8, 0.6, 0.9]
    spans = decode_bio_to_spans(label_ids, offsets, id2label, confs=confs)
    assert spans == [
        {'type': 'PER', 'start': 0, 'end': 6, 'score': 0.7},
        {'type': 'ORG', 'start': 7, 'end': 10, 'score': 0.9},
    ]


def test_decode_without_confs_has_no_score_bc():
    """confs 미입력 시 score 키 없음 — 기존 출력과 완전 동일(BC)."""
    label2id, id2label = build_label_maps()
    label_ids = [label2id['B-PER'], label2id['I-PER']]
    offsets = [(0, 3), (3, 6)]
    spans = decode_bio_to_spans(label_ids, offsets, id2label)
    assert spans == [{'type': 'PER', 'start': 0, 'end': 6}]
    assert 'score' not in spans[0]


def test_apply_thresholds_filters_below_cutoff():
    """score < 임계값 span 만 제거, 그 외(미대상 type·score 없음)는 통과."""
    preds = [[
        {'type': 'ORG', 'start': 0, 'end': 3, 'score': 0.95},   # 유지
        {'type': 'ORG', 'start': 5, 'end': 8, 'score': 0.40},   # 컷
        {'type': 'EMAIL', 'start': 9, 'end': 15, 'score': 0.1},  # 미대상→통과
        {'type': 'LOC', 'start': 16, 'end': 20},                # score없음→통과
    ]]
    out = apply_thresholds(preds, {'ORG': 0.7})
    kept = {(s['type'], s['start']) for s in out[0]}
    assert kept == {('ORG', 0), ('EMAIL', 9), ('LOC', 16)}


def _org(start, score, ttype='ORG'):
    return {'type': ttype, 'start': start, 'end': start + 2, 'score': score}


def test_fit_thresholds_recovers_precision():
    """저신뢰 FP 가 섞인 set 에서 fit 한 임계값이 P·R≥0.93 달성."""
    # gold: 문장마다 ORG 1개 (총 100 TP). pred: 정답 TP(score 0.95) +
    # 일부 문장에 저신뢰 FP(score 0.40) 30개.
    gold, pred = [], []
    for i in range(100):
        g = [{'type': 'ORG', 'start': 0, 'end': 2}]
        p = [_org(0, 0.95)]
        if i < 30:
            p.append(_org(5, 0.40))  # FP
        gold.append(g)
        pred.append(p)
    base = compute_offset_span_f1(gold, pred)['overall']
    assert base['precision'] < 0.93  # 임계값 없이는 미달

    thr = fit_thresholds(gold, pred, target_p=0.93, target_r=0.93)
    assert 0.40 <= thr['ORG'] <= 0.95  # FP 와 TP 사이에서 컷
    after = compute_offset_span_f1(gold, apply_thresholds(pred, thr))['overall']
    assert after['precision'] >= 0.93
    assert after['recall'] >= 0.93


def test_fit_thresholds_deterministic():
    """동일 입력 → 동일 임계값 (난수 없음)."""
    gold = [[{'type': 'ORG', 'start': 0, 'end': 2}] for _ in range(50)]
    pred = [[_org(0, 0.9), _org(5, 0.3)] for _ in range(50)]
    a = fit_thresholds(gold, pred)
    b = fit_thresholds(gold, pred)
    assert a == b


def test_save_load_roundtrip(tmp_path):
    """save→load 시 임계값 보존, meta 동봉 가능."""
    p = tmp_path / 'thresholds.json'
    thr = {'ORG': 0.72, 'LOC': 0.80, 'EVT': 0.85, 'PROD': 0.81}
    save_thresholds(str(p), thr, meta={'conf_key': 'conf_mean'})
    assert load_thresholds(str(p)) == thr
    payload = json.loads(p.read_text())
    assert payload['meta']['conf_key'] == 'conf_mean'
