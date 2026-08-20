"""`verify_labels` — 검증 정책이 원본 gold 를 지우지 못하게 막는다.

주입한 PII 와 원본 gold 는 증거의 성격이 다르다. 주입값은 우리가 무엇을
어디에 넣었는지 아니까 "LLM 이 못 찾았다" 가 곧 "주입이 어긋났다" 는 신호다.
원본 gold 는 사람이 붙인 정답이라 LLM 이 못 찾은 것은 **LLM 에 대한 증거**다.
구분 없이 `drop_span` 을 걸면 검증 모델의 recall 부족이 사람 주석을 지운다 —
EN 실측에서 gold NER 8,199 span 중 2,199 개(27%)가 그렇게 사라졌다.
"""
from ner.augmenters.pii.schema import Entity, Record
from ner.augmenters.pii.verifier import PIIVerifier, VerifyPolicy

PII_ONLY = ['EMAIL', 'PHONE', 'ID_NUM', 'CREDIT_CARD']


class StubLabeler:
    """지정한 span 만 찾아내는 라벨러 — 검증 모델의 recall 구멍을 흉내낸다."""

    def __init__(self, found):
        self._found = found

    def label_spans(self, text, split=True):
        return list(self._found)


def _record():
    return Record(
        text='Reach Mary Moore of Acme Corp at m@x.com on May 1.',
        entities=[
            Entity(label='PER', start_char=6, end_char=16, text='Mary Moore'),
            Entity(label='ORG', start_char=20, end_char=29, text='Acme Corp'),
            Entity(label='DAT', start_char=44, end_char=49, text='May 1'),
            Entity(label='EMAIL', start_char=33, end_char=40, text='m@x.com'),
        ],
    )


def test_scoped_verification_keeps_gold_the_model_missed():
    """라벨러가 EMAIL 만 찾아도 PER·ORG·DAT 은 살아남는다."""
    verifier = PIIVerifier(
        StubLabeler([{'text': 'm@x.com', 'type': 'EMAIL'}]),
        policy=VerifyPolicy.DROP_SPAN,
        verify_labels=PII_ONLY,
    )
    result = verifier.verify(_record())
    labels = sorted(e.label for e in result.record.entities)
    assert labels == ['DAT', 'EMAIL', 'ORG', 'PER']
    assert sorted(e.label for e in result.exempt) == ['DAT', 'ORG', 'PER']
    assert [e.label for e in result.confirmed] == ['EMAIL']
    assert result.missed == []


def test_scoped_verification_still_drops_bad_injections():
    """범위 안 라벨은 확인되지 않으면 그대로 버린다 — 검증이 살아 있다."""
    verifier = PIIVerifier(
        StubLabeler([{'text': 'Mary Moore', 'type': 'PER'}]),
        policy=VerifyPolicy.DROP_SPAN,
        verify_labels=PII_ONLY,
    )
    result = verifier.verify(_record())
    assert 'EMAIL' not in [e.label for e in result.record.entities]
    assert [e.label for e in result.missed] == ['EMAIL']


def test_unscoped_verification_is_unchanged():
    """기본값(None)은 기존 동작 — ja·vi·ko 의 BC 를 지킨다."""
    verifier = PIIVerifier(
        StubLabeler([{'text': 'm@x.com', 'type': 'EMAIL'}]),
        policy=VerifyPolicy.DROP_SPAN,
    )
    result = verifier.verify(_record())
    assert [e.label for e in result.record.entities] == ['EMAIL']
    assert sorted(e.label for e in result.missed) == ['DAT', 'ORG', 'PER']
    assert result.exempt == []


def test_keep_all_policy_is_unaffected_by_scope():
    verifier = PIIVerifier(
        StubLabeler([]), policy=VerifyPolicy.KEEP_ALL,
        verify_labels=PII_ONLY,
    )
    result = verifier.verify(_record())
    assert len(result.record.entities) == 4


def test_drop_record_only_reacts_to_in_scope_issues():
    """범위 밖 라벨을 못 찾았다고 레코드를 통째로 버리면 안 된다."""
    verifier = PIIVerifier(
        StubLabeler([{'text': 'm@x.com', 'type': 'EMAIL'}]),
        policy=VerifyPolicy.DROP_RECORD,
        verify_labels=PII_ONLY,
    )
    assert verifier.verify(_record()).dropped is False


def test_report_records_the_scope_and_exempt_count():
    verifier = PIIVerifier(
        StubLabeler([{'text': 'm@x.com', 'type': 'EMAIL'}]),
        policy=VerifyPolicy.DROP_SPAN,
        verify_labels=PII_ONLY,
    )
    _, report = verifier.verify_dataset([_record()])
    assert report['verify_labels'] == sorted(PII_ONLY)
    assert report['exempt_count'] == 3
    assert report['per_label']['PER']['exempt'] == 1


def test_report_marks_legacy_runs_as_unscoped():
    verifier = PIIVerifier(StubLabeler([]), policy=VerifyPolicy.KEEP_ALL)
    _, report = verifier.verify_dataset([_record()])
    assert report['verify_labels'] is None
    assert report['exempt_count'] == 0
