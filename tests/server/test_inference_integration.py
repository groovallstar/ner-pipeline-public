"""모델 통합 테스트 — 실제 /data/ner/{ja,ko,vi,en} 모델 의존(로컬 전용).

/data 는 gitignore 라 CI 에서 재현 불가하므로, 모델 디렉토리가 없으면
`pytest.skip`(약화 아님 — 사유 출력). 핵심은 모델 품질이 아니라 **offset
정합성**: 반환된 어떤 span 도 `text[start_char:end_char] == span['text']`.

네 언어를 같은 축으로 태운다. 계약 테스트(`test_api.py`)는 stub registry 라
모델을 안 부르므로, 토크나이저 분기·청킹·배치화가 언어별로 갈리는 자리는
여기서만 잡힌다 — ja 만 slow 토크나이저 특례이고 나머지 셋은 fast 경로다
(en 은 roberta-base).

`chunking._SENT_RE` 의 경계 문자 집합에는 언어 분기가 없고, 그 집합에 ASCII
마침표가 없다. 그래서 마침표로 끝나는 산문은 ko 든 en 이든 문장 분할이 아니라
단어 경계 폴백(`_char_windows`)을 탄다.
"""

import json
import os
import unicodedata

import pytest

from ner.classifier.data_utils import load_jsonl
from ner.metrics.span_metrics import compute_offset_span_f1
from server.chunking import split_for_length
from server.config import ServerConfig
from server.inference import LangModel, ModelRegistry, _load_tokenizer

_CONFIG = ServerConfig()
_JA_DIR = _CONFIG.model_dir('ja')
_VI_DIR = _CONFIG.model_dir('vi')
_KO_DIR = _CONFIG.model_dir('ko')
_EN_DIR = _CONFIG.model_dir('en')
_JA_TEST = os.path.join(_CONFIG.model_root, 'ja', 'data', 'test.jsonl')
_VI_TEST = os.path.join(_CONFIG.model_root, 'vi', 'data', 'test.jsonl')
_KO_TEST = os.path.join(_CONFIG.model_root, 'ko', 'data', 'test.jsonl')
_EN_TEST = os.path.join(_CONFIG.model_root, 'en', 'data', 'test.jsonl')
_JA_METRICS = os.path.join(_CONFIG.model_root, 'ja', 'metrics.json')
_KO_METRICS = os.path.join(_CONFIG.model_root, 'ko', 'metrics.json')
_EN_METRICS = os.path.join(_CONFIG.model_root, 'en', 'metrics.json')
_JA_PARITY_READY = (os.path.isdir(_JA_DIR) and os.path.isfile(_JA_TEST)
                    and os.path.isfile(_JA_METRICS))
_KO_PARITY_READY = (os.path.isdir(_KO_DIR) and os.path.isfile(_KO_TEST)
                    and os.path.isfile(_KO_METRICS))
_EN_PARITY_READY = (os.path.isdir(_EN_DIR) and os.path.isfile(_EN_TEST)
                    and os.path.isfile(_EN_METRICS))
_JA_CHUNK_READY = os.path.isdir(_JA_DIR) and os.path.isfile(_JA_TEST)
_VI_CHUNK_READY = os.path.isdir(_VI_DIR) and os.path.isfile(_VI_TEST)
_KO_CHUNK_READY = os.path.isdir(_KO_DIR) and os.path.isfile(_KO_TEST)
_EN_CHUNK_READY = os.path.isdir(_EN_DIR) and os.path.isfile(_EN_TEST)


def _assert_offsets_consistent(model, text):
    for ent in model.predict(text):
        assert text[ent['start_char']:ent['end_char']] == ent['text']


def _build_long_doc(rows):
    """gold row 들을 이어붙여 장문 + 글로벌 offset gold 엔티티를 구성.

    row 사이에 개행을 넣어 문장 경계를 보존하고, 각 엔티티 offset 은 누적
    길이(base)로 원문 글로벌 위치로 보정한다.
    """
    text = ''
    ents = []
    for row in rows:
        base = len(text)
        for e in row['entities']:
            ents.append({'label': e['label'],
                         'start': base + e['start_char'],
                         'end': base + e['end_char']})
        text += row['text'] + '\n'
    return text, ents


def _chunk_spans(text, tok, max_length):
    """(start, end) 청크 구간 리스트 — 엔티티 포함 판정용."""
    return [(base, base + len(sub))
            for sub, base in split_for_length(text, tok, max_length)]


def _assert_latter_half_recall(model, text):
    """예측 엔티티가 후반 50% 영역에도 존재하고 offset 이 원문과 일치.

    장문이 청크로 쪼개진 뒤 후반 청크의 엔티티가 누락되지 않았다는 증거 —
    offset 정합성만으로는 잡지 못하는 recall 측면을 검증한다.
    """
    ents = model.predict(text)
    assert ents
    latter = [e for e in ents if e['start_char'] > len(text) * 0.5]
    assert latter, 'no entity recovered in the latter half (late-chunk drop?)'
    for ent in ents:
        assert text[ent['start_char']:ent['end_char']] == ent['text']


def _assert_batched_matches_per_chunk(model, text):
    """배치 forward(predict)가 chunk별 단건 forward 와 동일한 span 을 낸다.

    순차 chunk 경로가 배치 경로로 대체됐으므로 그 parity 를 회귀로 고정한다.
    같은 장문을 (i) `predict` 한 번(모든 chunk 를 `[K, max_length]` 배치로 1
    forward) (ii) chunk 별 단건 `predict`(각 substring 은 1 chunk → 단건
    forward) 후 글로벌 offset shift 로 각각 구해 비교한다. label·offset·
    표면형은 정확히, score 는 GPU 커널 비결정성 대비 1e-5 허용.

    추론은 fp32 로만 돌아 배치(B>1)·단건(B=1)이 같은 커널을 타므로,
    배치화 자체(pad/stack/순서·offset 복원)의 등가만 본다.
    apply_threshold=False 로 임계값 경로도 배제한다.
    """
    chunks = split_for_length(text, model.tokenizer, model.max_length)
    assert len(chunks) > 1  # 강제 분할 — 배치 경로가 실제로 작동
    batched = model.predict(text, apply_threshold=False)
    per_chunk = []
    for sub, base in chunks:
        for e in model.predict(sub, apply_threshold=False):
            per_chunk.append({'label': e['label'],
                              'start_char': e['start_char'] + base,
                              'end_char': e['end_char'] + base,
                              'text': e['text'],
                              'score': e['score']})
    assert batched  # 엔티티가 있어야 비교가 공허하지 않음
    assert len(batched) == len(per_chunk)
    for b, r in zip(batched, per_chunk):
        assert (b['label'], b['start_char'], b['end_char'], b['text']) == \
               (r['label'], r['start_char'], r['end_char'], r['text'])
        assert b['score'] == pytest.approx(r['score'], abs=1e-5)


@pytest.mark.skipif(not os.path.isdir(_JA_DIR),
                    reason=f'ja model dir not present: {_JA_DIR}')
def test_ja_offset_consistency():
    """ja: 반환 span offset 이 원문 표면형과 일치."""
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    _assert_offsets_consistent(model, '織田信長は東京都千代田区に住んでいた。')


@pytest.mark.skipif(not os.path.isdir(_VI_DIR),
                    reason=f'vi model dir not present: {_VI_DIR}')
def test_vi_offset_consistency():
    """vi(PhoBERT): 반환 span offset 이 원문 표면형과 일치."""
    model = LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                      _CONFIG.max_length)
    _assert_offsets_consistent(model, 'Hà Nội là thủ đô của Việt Nam.')


@pytest.mark.skipif(not os.path.isdir(_KO_DIR),
                    reason=f'ko model dir not present: {_KO_DIR}')
def test_ko_offset_consistency():
    """ko(KoELECTRA): 반환 span offset 이 원문 표면형과 일치."""
    model = LangModel('ko', _KO_DIR, _CONFIG.thresholds_path('ko'),
                      _CONFIG.max_length)
    _assert_offsets_consistent(model, '김민준은 서울 강남구에 살고 있었다.')


@pytest.mark.skipif(not os.path.isdir(_EN_DIR),
                    reason=f'en model dir not present: {_EN_DIR}')
def test_en_offset_consistency():
    """en(roberta-base): 반환 span offset 이 원문 표면형과 일치."""
    model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                      _CONFIG.max_length)
    _assert_offsets_consistent(
        model, 'Barack Obama was born in Hawaii in 1961.')


@pytest.mark.skipif(not os.path.isdir(_VI_DIR),
                    reason=f'vi model dir not present: {_VI_DIR}')
def test_vi_graceful_raw_when_no_thresholds():
    """vi 는 thresholds.json 부재 → raw(abstention 미적용)로 동작."""
    model = LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                      _CONFIG.max_length)
    assert model.has_thresholds is False


@pytest.mark.skipif(not os.path.isdir(_KO_DIR),
                    reason=f'ko model dir not present: {_KO_DIR}')
def test_ko_graceful_raw_when_no_thresholds():
    """ko 도 thresholds.json 없이 출하돼 raw 로 동작(vi 와 같은 경로)."""
    model = LangModel('ko', _KO_DIR, _CONFIG.thresholds_path('ko'),
                      _CONFIG.max_length)
    assert model.has_thresholds is False


@pytest.mark.skipif(not os.path.isdir(_EN_DIR),
                    reason=f'en model dir not present: {_EN_DIR}')
def test_en_graceful_raw_when_no_thresholds():
    """en 도 thresholds.json 없이 출하돼 raw 로 동작.

    vi·ko 와 **로드 경로만** 같다. 감지 경로는 다르다 — vi 는 고위 특이도
    양성 신호로 도달하는데 en 은 신호의 부재로 도달하므로, raw 출력에
    라우팅 오차가 곱해진다.
    """
    model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                      _CONFIG.max_length)
    assert model.has_thresholds is False
    assert model.thresholds == {}


@pytest.mark.skipif(not os.path.isdir(_EN_DIR),
                    reason=f'en model dir not present: {_EN_DIR}')
def test_en_health_reports_raw_serving():
    """`/health` 가 en 을 loaded·thresholds False 로 보고한다.

    `has_thresholds` 를 객체 속성이 아니라 **출하되는 응답**으로 단언해,
    "en 은 raw 로 뜬다" 를 내부 사실이 아니라 소비자 계약으로 만든다.
    """
    model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                      _CONFIG.max_length)
    health = ModelRegistry({'en': model}).health()
    assert health['langs']['en'] == {'loaded': True, 'thresholds': False}


@pytest.mark.skipif(not os.path.isdir(_JA_DIR),
                    reason=f'ja model dir not present: {_JA_DIR}')
def test_ja_long_input_chunk_offsets():
    """max_length 초과 입력: chunk 분할 후에도 span offset·recall 보존.

    chunk 가 실제로 쪼개지는지 확인한 뒤(>1), 후반 50% 영역에도 엔티티가
    회수되고(late-chunk 누락 없음) 모든 span 의 글로벌 offset 이 원문과
    일치하는지 검증한다 — auto-chunk offset 보정 + recall.
    """
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    text = '東京都に住む織田信長は安土城を築いた。' * 40
    chunks = split_for_length(text, model.tokenizer, model.max_length)
    assert len(chunks) > 1
    _assert_latter_half_recall(model, text)


@pytest.mark.skipif(not _VI_CHUNK_READY,
                    reason='vi model/test not present')
def test_vi_long_input_chunk_offsets():
    """vi(PhoBERT): max_length 초과 입력도 후반 청크 엔티티를 회수하고
    글로벌 offset 이 원문과 일치(auto-chunk offset 보정 + recall)."""
    model = LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_VI_TEST) if r['entities']][:30]
    text, _ = _build_long_doc(rows)
    chunks = split_for_length(text, model.tokenizer, model.max_length)
    assert len(chunks) > 1
    _assert_latter_half_recall(model, text)


@pytest.mark.skipif(not _KO_CHUNK_READY,
                    reason='ko model/test not present')
def test_ko_long_input_chunk_offsets():
    """ko: max_length 초과 입력도 후반 청크 엔티티를 회수하고 글로벌 offset
    이 원문과 일치.

    한국어는 문장 경계 정규식이 ASCII 마침표를 안 잡아 단어 경계 폴백으로
    쪼개진다 — ja·vi 와 다른 분할 경로라 offset 보정도 따로 확인한다.
    """
    model = LangModel('ko', _KO_DIR, _CONFIG.thresholds_path('ko'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_KO_TEST) if r['entities']][:30]
    text, _ = _build_long_doc(rows)
    chunks = split_for_length(text, model.tokenizer, model.max_length)
    assert len(chunks) > 1
    _assert_latter_half_recall(model, text)


@pytest.mark.skipif(not _EN_CHUNK_READY,
                    reason='en model/test not present')
def test_en_long_input_chunk_offsets():
    """en: max_length 초과 입력도 후반 청크 엔티티를 회수하고 글로벌
    offset 이 원문과 일치.

    영어 산문은 ASCII 마침표로 끝나는데 경계 정규식이 그것을 안 잡으므로,
    ko 와 같은 단어 경계 폴백을 탄다.
    """
    model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_EN_TEST) if r['entities']][:30]
    text, _ = _build_long_doc(rows)
    chunks = split_for_length(text, model.tokenizer, model.max_length)
    assert len(chunks) > 1
    _assert_latter_half_recall(model, text)


@pytest.mark.skipif(not _JA_CHUNK_READY,
                    reason='ja model/test not present')
def test_ja_batched_chunks_match_per_chunk():
    """ja: multi-chunk 배치 predict 가 chunk별 단건 forward 와 동일(parity)."""
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_JA_TEST) if r['entities']][:30]
    text, _ = _build_long_doc(rows)
    _assert_batched_matches_per_chunk(model, text)


@pytest.mark.skipif(not _VI_CHUNK_READY,
                    reason='vi model/test not present')
def test_vi_batched_chunks_match_per_chunk():
    """vi: multi-chunk 배치 predict 가 chunk별 단건 forward 와 동일(parity)."""
    model = LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_VI_TEST) if r['entities']][:30]
    text, _ = _build_long_doc(rows)
    _assert_batched_matches_per_chunk(model, text)


@pytest.mark.skipif(not _KO_CHUNK_READY,
                    reason='ko model/test not present')
def test_ko_batched_chunks_match_per_chunk():
    """ko: multi-chunk 배치 predict 가 chunk별 단건 forward 와 동일(parity)."""
    model = LangModel('ko', _KO_DIR, _CONFIG.thresholds_path('ko'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_KO_TEST) if r['entities']][:30]
    text, _ = _build_long_doc(rows)
    _assert_batched_matches_per_chunk(model, text)


@pytest.mark.skipif(not _EN_CHUNK_READY,
                    reason='en model/test not present')
def test_en_batched_chunks_match_per_chunk():
    """en: multi-chunk 배치 predict 가 chunk별 단건 forward 와 동일."""
    model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_EN_TEST) if r['entities']][:30]
    text, _ = _build_long_doc(rows)
    _assert_batched_matches_per_chunk(model, text)


@pytest.mark.skipif(not _JA_CHUNK_READY,
                    reason='ja model/test not present')
def test_ja_gold_entities_never_straddle_chunk_boundary():
    """ja: 실 test gold 엔티티 전수가 청크 경계를 가로지르지 않는다.

    전 row 를 한 장문으로 이어 강제 분할(>1 청크)한 뒤, 각 gold 엔티티가
    정확히 한 청크에 온전히 들어감을 확인한다 — 엔티티가 경계를 넘으면
    모델에 잘려 보여 recall 이 떨어지므로, 이 불변식이 recall 보존을 보장한다.
    """
    tok = _load_tokenizer(_JA_DIR, 'ja')
    text, ents = _build_long_doc(load_jsonl(_JA_TEST))
    spans = _chunk_spans(text, tok, _CONFIG.max_length)
    assert len(spans) > 1 and ents  # 강제 분할·엔티티 존재 — 테스트 비공허
    for e in ents:
        assert any(cs <= e['start'] and e['end'] <= ce for cs, ce in spans), \
            f"ja entity {text[e['start']:e['end']]!r} straddles a boundary"


@pytest.mark.skipif(not _VI_CHUNK_READY,
                    reason='vi model/test not present')
def test_vi_gold_entities_never_straddle_chunk_boundary():
    """vi: 실 test gold 엔티티 전수가 청크 경계를 가로지르지 않는다(recall 보존)."""
    tok = _load_tokenizer(_VI_DIR, 'vi')
    text, ents = _build_long_doc(load_jsonl(_VI_TEST))
    spans = _chunk_spans(text, tok, _CONFIG.max_length)
    assert len(spans) > 1 and ents
    for e in ents:
        assert any(cs <= e['start'] and e['end'] <= ce for cs, ce in spans), \
            f"vi entity {text[e['start']:e['end']]!r} straddles a boundary"


@pytest.mark.skipif(not _KO_CHUNK_READY,
                    reason='ko model/test not present')
def test_ko_gold_entities_never_straddle_chunk_boundary():
    """ko: 실 test gold 엔티티 전수가 청크 경계를 가로지르지 않는다.

    한국어는 단어 경계 폴백으로 쪼개지므로 이 불변식이 ja·vi 의 문장 분할과
    다른 코드 경로에서 성립함을 확인한다 — 폴백이 어절 중간을 자르면 엔티티가
    잘려 recall 이 떨어진다.
    """
    tok = _load_tokenizer(_KO_DIR, 'ko')
    text, ents = _build_long_doc(load_jsonl(_KO_TEST))
    spans = _chunk_spans(text, tok, _CONFIG.max_length)
    assert len(spans) > 1 and ents
    for e in ents:
        assert any(cs <= e['start'] and e['end'] <= ce for cs, ce in spans), \
            f"ko entity {text[e['start']:e['end']]!r} straddles a boundary"


@pytest.mark.skipif(not _EN_CHUNK_READY,
                    reason='en model/test not present')
def test_en_gold_entities_never_straddle_chunk_boundary():
    """en: 실 test gold 엔티티 전수가 청크 경계를 가로지르지 않는다.

    영어도 ko 와 같은 단어 경계 폴백을 타므로, 폴백이 어절 중간을 자르면
    엔티티가 잘려 recall 이 떨어진다.
    """
    tok = _load_tokenizer(_EN_DIR, 'en')
    text, ents = _build_long_doc(load_jsonl(_EN_TEST))
    spans = _chunk_spans(text, tok, _CONFIG.max_length)
    assert len(spans) > 1 and ents
    for e in ents:
        assert any(cs <= e['start'] and e['end'] <= ce for cs, ce in spans), \
            f"en entity {text[e['start']:e['end']]!r} straddles a boundary"


def _overall_f1(model, rows, apply_threshold):
    """test rows 에 대한 strict overall F1 — 서버 predict 경로로 계산.

    gold/pred 를 metrics 모듈이 요구하는 {type,start,end} 형식으로 변환한다.
    언어 무관 — 넘긴 model·rows 가 언어를 정한다.
    """
    gold_list, pred_list = [], []
    for row in rows:
        gold_list.append([
            {'type': e['label'], 'start': e['start_char'],
             'end': e['end_char']}
            for e in row['entities']])
        pred_list.append([
            {'type': s['label'], 'start': s['start_char'],
             'end': s['end_char']}
            for s in model.predict(row['text'],
                                   apply_threshold=apply_threshold)])
    return compute_offset_span_f1(gold_list, pred_list)['overall']


@pytest.mark.skipif(not _JA_PARITY_READY,
                    reason='ja model/test/metrics not all present')
def test_ja_parity_operating_point():
    """abstention on → 서버 operating-point F1 이 metrics.json 과 일치.

    eval 스크립트를 import 하지 않고, 기록된 운영점(데이터)과 대조해 parity
    를 검증한다(리팩터 중인 코드에 비의존).
    """
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    rows = load_jsonl(_JA_TEST)
    expected = json.load(open(_JA_METRICS, encoding='utf-8'))[
        'abstention']['overall_operating']
    got = _overall_f1(model, rows, apply_threshold=True)
    assert got['f1'] == pytest.approx(expected['f1'], abs=1e-6)
    assert got['precision'] == pytest.approx(expected['precision'], abs=1e-6)
    assert got['recall'] == pytest.approx(expected['recall'], abs=1e-6)


@pytest.mark.skipif(not _JA_PARITY_READY,
                    reason='ja model/test/metrics not all present')
def test_ja_parity_baseline_raw():
    """abstention off → raw baseline F1 이 metrics.json baseline 과 일치."""
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    rows = load_jsonl(_JA_TEST)
    expected = json.load(open(_JA_METRICS, encoding='utf-8'))[
        'abstention']['overall_baseline']
    got = _overall_f1(model, rows, apply_threshold=False)
    assert got['f1'] == pytest.approx(expected['f1'], abs=1e-6)


@pytest.mark.skipif(not _KO_PARITY_READY,
                    reason='ko model/test/metrics not all present')
def test_ko_parity_baseline_raw():
    """ko: 서버 predict 의 raw F1 이 배포 metrics.json 과 일치.

    ko 는 thresholds 없이 출하돼 raw 가 운영점이므로 대조 대상이 하나다
    (ja 는 abstention on/off 둘). `overall_strict` 는 배포 run 자신을 승격한
    원장이라, 이 대조는 재현이 아니라 **패키지가 자기가 나온 run 과 어긋나지
    않는가**를 본다 — 모델·토크나이저·디코드 경로가 바뀌면 여기서 갈린다.
    """
    model = LangModel('ko', _KO_DIR, _CONFIG.thresholds_path('ko'),
                      _CONFIG.max_length)
    rows = load_jsonl(_KO_TEST)
    expected = json.load(open(_KO_METRICS, encoding='utf-8'))['overall_strict']
    got = _overall_f1(model, rows, apply_threshold=False)
    assert got['f1'] == pytest.approx(expected['f1'], abs=1e-6)
    assert got['precision'] == pytest.approx(expected['precision'], abs=1e-6)
    assert got['recall'] == pytest.approx(expected['recall'], abs=1e-6)
    assert got['support'] == expected['support']


@pytest.mark.skipif(not _EN_PARITY_READY,
                    reason='en model/test/metrics not all present')
def test_en_parity_baseline_raw():
    """en: 서버 predict 의 raw F1 이 배포 metrics.json 과 일치.

    en 은 thresholds 없이 출하돼 raw 가 운영점이므로 대조 대상이 하나다
    (ko 와 같은 모양). 양변 모두 fp32 라 정밀도 델타가 없다 —
    `metrics.json` 의 `precision: "bf16"` 은 학습 혼합정밀도 기록이지 평가
    정밀도가 아니다.
    """
    model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                      _CONFIG.max_length)
    rows = load_jsonl(_EN_TEST)
    expected = json.load(open(_EN_METRICS, encoding='utf-8'))['overall_strict']
    got = _overall_f1(model, rows, apply_threshold=False)
    assert got['f1'] == pytest.approx(expected['f1'], abs=1e-6)
    assert got['precision'] == pytest.approx(expected['precision'], abs=1e-6)
    assert got['recall'] == pytest.approx(expected['recall'], abs=1e-6)
    assert got['support'] == expected['support']


def _mixed_length_texts(rows):
    """짧은 단건 + 장문(multi-chunk)을 섞은 텍스트 리스트(B≥2)."""
    short = [r['text'] for r in rows[:4]]
    long_doc, _ = _build_long_doc(rows)
    return short + [long_doc]


def _assert_spans_equal(got, exp):
    """label·offset·표면형은 정확히, score 는 1e-5 허용(배치 커널 드리프트).

    같은 fp32 라도 GPU matmul 은 배치 크기에 따라 커널이 달라 score 가
    1e-7 수준으로 흔들린다 — 결정(label·offset)은 불변이어야 한다.
    """
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        assert (g['label'], g['start_char'], g['end_char'], g['text']) == \
               (e['label'], e['start_char'], e['end_char'], e['text'])
        assert g['score'] == pytest.approx(e['score'], abs=1e-5)


def _assert_predict_many_matches_single(model, texts):
    """predict_many(배치)가 텍스트별 단건 predict 와 동일 — 배치화 등가.

    추론은 fp32 로만 돌아 배치(B>1)·단건(B=1)이 같은 정밀도를 쓰므로,
    cross-text 묶음→복원(입력 순서·글로벌 offset)이 단건 순차와 동일함을
    검증한다(label·offset 정확, score 는 커널 드리프트 1e-5 허용).
    """
    batched = model.predict_many(texts, apply_threshold=False)
    single = [model.predict(t, apply_threshold=False) for t in texts]
    assert len(batched) == len(texts)
    for b, s in zip(batched, single):
        _assert_spans_equal(b, s)


@pytest.mark.skipif(not _JA_CHUNK_READY, reason='ja model/test not present')
def test_ja_predict_many_matches_single():
    """ja: cross-text 배치(혼합 길이)가 단건 순차와 동일(배치화 등가)."""
    model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_JA_TEST) if r['entities']][:30]
    _assert_predict_many_matches_single(model, _mixed_length_texts(rows))


@pytest.mark.skipif(not _VI_CHUNK_READY, reason='vi model/test not present')
def test_vi_predict_many_matches_single():
    """vi: cross-text 배치(혼합 길이)가 단건 순차와 동일(배치화 등가)."""
    model = LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_VI_TEST) if r['entities']][:30]
    _assert_predict_many_matches_single(model, _mixed_length_texts(rows))


@pytest.mark.skipif(not _KO_CHUNK_READY, reason='ko model/test not present')
def test_ko_predict_many_matches_single():
    """ko: cross-text 배치(혼합 길이)가 단건 순차와 동일(배치화 등가)."""
    model = LangModel('ko', _KO_DIR, _CONFIG.thresholds_path('ko'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_KO_TEST) if r['entities']][:30]
    _assert_predict_many_matches_single(model, _mixed_length_texts(rows))


@pytest.mark.skipif(not _EN_CHUNK_READY, reason='en model/test not present')
def test_en_predict_many_matches_single():
    """en: cross-text 배치(혼합 길이)가 단건 순차와 동일(배치화 등가)."""
    model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                      _CONFIG.max_length)
    rows = [r for r in load_jsonl(_EN_TEST) if r['entities']][:30]
    _assert_predict_many_matches_single(model, _mixed_length_texts(rows))


@pytest.mark.skipif(
    not (_JA_CHUNK_READY and _VI_CHUNK_READY and _KO_CHUNK_READY
         and _EN_CHUNK_READY),
    reason='ja+vi+ko+en models/test not all present')
def test_predict_batch_mixed_lang_matches_single():
    """이질 배치(ja·vi·ko·en 혼합·인터리브)가 순서·lang·offset 1:1로 단건과
    동일.

    registry.predict_batch 가 언어별로 묶어 forward 한 뒤 입력 순서로 복원
    하므로, 추론이 fp32 라 각 항목이 단건 predict 와 정확히 일치해야
    한다 — 묶음/복원에서 순서·언어가 섞이면 깨진다. 네 언어를 인터리브하는
    것은 묶음이 둘일 때보다 복원이 어긋날 자리가 많아서다.
    """
    ja_model = LangModel('ja', _JA_DIR, _CONFIG.thresholds_path('ja'),
                         _CONFIG.max_length)
    vi_model = LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                         _CONFIG.max_length)
    ko_model = LangModel('ko', _KO_DIR, _CONFIG.thresholds_path('ko'),
                         _CONFIG.max_length)
    en_model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                         _CONFIG.max_length)
    registry = ModelRegistry({'ja': ja_model, 'vi': vi_model,
                              'ko': ko_model, 'en': en_model})
    ja_rows = [r for r in load_jsonl(_JA_TEST) if r['entities']][:3]
    vi_rows = [r for r in load_jsonl(_VI_TEST) if r['entities']][:3]
    ko_rows = [r for r in load_jsonl(_KO_TEST) if r['entities']][:3]
    en_rows = [r for r in load_jsonl(_EN_TEST) if r['entities']][:3]
    texts, langs = [], []
    for jr, vr, kr, er in zip(ja_rows, vi_rows, ko_rows,
                              en_rows):  # ja·vi·ko·en 인터리브
        texts.append(jr['text'])
        langs.append('ja')
        texts.append(vr['text'])
        langs.append('vi')
        texts.append(kr['text'])
        langs.append('ko')
        texts.append(er['text'])
        langs.append('en')
    out = registry.predict_batch(texts, langs, apply_threshold=False)
    assert len(out) == len(texts)
    assert set(langs) == {'ja', 'vi', 'ko', 'en'}  # 네 묶음이 실제로 만들어졌다
    for i, (t, lang) in enumerate(zip(texts, langs)):
        _assert_spans_equal(
            out[i], registry.predict(t, lang, apply_threshold=False))


@pytest.mark.skipif(not _VI_CHUNK_READY, reason='vi model/test not present')
def test_vi_nfd_input_matches_nfc():
    """NFD(분해형) 베트남어 입력이 NFC 와 같은 엔티티를 낸다.

    회귀 가드: NFD 입력은 결합부호가 별도 토큰이 돼 엔티티가 소실·절단됐다.
    predict 가 입력을 NFC 로 정규화해 두 형태가 동일 결과를 내게 한다.
    """
    model = LangModel('vi', _VI_DIR, _CONFIG.thresholds_path('vi'),
                      _CONFIG.max_length)
    text = 'Hà Nội là thủ đô của Việt Nam.'
    nfc = [(s['label'], s['start_char'], s['end_char'], s['text'])
           for s in model.predict(unicodedata.normalize('NFC', text))]
    nfd = [(s['label'], s['start_char'], s['end_char'], s['text'])
           for s in model.predict(unicodedata.normalize('NFD', text))]
    assert nfc == nfd                        # 정규화로 두 형태가 일치
    assert any(lbl == 'LOC' for lbl, *_ in nfc)  # 엔티티가 실제로 잡힘


@pytest.mark.skipif(not os.path.isdir(_KO_DIR),
                    reason=f'ko model dir not present: {_KO_DIR}')
def test_ko_nfd_input_matches_nfc():
    """NFD(자모 분해형) 한국어 입력이 NFC 와 같은 엔티티를 낸다.

    한글 음절은 NFD 로 초·중·종성이 갈려 길이가 배 이상 늘어난다(`김` 1자 →
    3자). 정규화 없이 들어가면 토크나이저가 자모를 낱개로 보고 offset 도
    분해형 기준이라 원문 슬라이스와 어긋난다. predict 가 입력을 NFC 로
    정규화하므로 두 형태가 같은 결과를 내야 한다 — vi 결합부호와 같은 가드가
    한글에도 필요한 이유다.
    """
    model = LangModel('ko', _KO_DIR, _CONFIG.thresholds_path('ko'),
                      _CONFIG.max_length)
    text = '김민준은 서울 강남구 삼성전자에서 갤럭시를 샀다.'
    assert len(unicodedata.normalize('NFD', text)) > len(text)  # 실제 분해
    nfc = [(s['label'], s['start_char'], s['end_char'], s['text'])
           for s in model.predict(unicodedata.normalize('NFC', text))]
    nfd = [(s['label'], s['start_char'], s['end_char'], s['text'])
           for s in model.predict(unicodedata.normalize('NFD', text))]
    assert nfc == nfd                        # 정규화로 두 형태가 일치
    assert any(lbl == 'PER' for lbl, *_ in nfc)  # 엔티티가 실제로 잡힘


@pytest.mark.skipif(not os.path.isdir(_EN_DIR),
                    reason=f'en model dir not present: {_EN_DIR}')
def test_en_nfd_input_matches_nfc():
    """NFD(분해형) 영어 입력이 NFC 와 같은 엔티티를 낸다.

    영어라고 ASCII 만 오는 것이 아니다 — 차용 인명·지명에 붙은 결합부호가
    NFD 로 갈리면 토크나이저가 낱개로 보고 offset 도 분해형 기준이라 원문
    슬라이스와 어긋난다. vi 결합부호·ko 자모와 같은 가드가 필요한 이유다.
    """
    model = LangModel('en', _EN_DIR, _CONFIG.thresholds_path('en'),
                      _CONFIG.max_length)
    text = 'Renée Fleming performed in Zürich with the Orchestre National.'
    assert len(unicodedata.normalize('NFD', text)) > len(text)  # 실제 분해
    nfc = [(s['label'], s['start_char'], s['end_char'], s['text'])
           for s in model.predict(unicodedata.normalize('NFC', text))]
    nfd = [(s['label'], s['start_char'], s['end_char'], s['text'])
           for s in model.predict(unicodedata.normalize('NFD', text))]
    assert nfc == nfd                        # 정규화로 두 형태가 일치
    assert nfc, 'expected at least one entity'
