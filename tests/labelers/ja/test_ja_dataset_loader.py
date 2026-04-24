"""Stockmark JA loader — LABEL_CORRECTIONS 정합 테스트."""
from labelers.ja.dataset_loader import JapaneseDatasetLoader


class _MockRow(dict):
    """HF 데이터셋 row 모사 (dict 접근 + .get)."""


def _rows():
    """LABEL_CORRECTIONS 적용 대상·비대상을 각 1건씩 포함하는 고정 시퀀스."""
    return [
        _MockRow(
            curid='2746825',
            text='2015年4月の人事異動で、地元NHK仙台放送局へ異動。',
            entities=[
                {'name': 'NHK', 'type': 'その他の組織名', 'span': [16, 19]},
                {'name': '仙台放送局', 'type': '施設名', 'span': [19, 24]},
            ],
        ),
        _MockRow(
            curid='2919391',
            text='香港政府から無料放送免許の申請が却下されたため',
            entities=[
                {'name': '香港政府', 'type': '法人名', 'span': [43, 47]},
            ],
        ),
        _MockRow(
            curid='2942700',
            text='コロラド・メサ大学とセントメアリー病院が主要な雇用主',
            entities=[
                {'name': 'コロラド・メサ大学', 'type': '法人名', 'span': [28, 37]},
                {'name': 'セントメアリー病院', 'type': '法人名', 'span': [38, 47]},
            ],
        ),
        _MockRow(
            curid='999999',
            text='関係のない文です。',
            entities=[{'name': '関係', 'type': '人名', 'span': [0, 2]}],
        ),
    ]


class TestLabelCorrections:
    def test_nhk_org_to_corp(self):
        recs = JapaneseDatasetLoader._to_records(_rows())
        nhk = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '2746825' and g['text'] == 'NHK'
        ]
        assert nhk and nhk[0]['type'] == '法人名'

    def test_government_corp_to_pol(self):
        recs = JapaneseDatasetLoader._to_records(_rows())
        gov = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '2919391' and g['text'] == '香港政府'
        ]
        assert gov and gov[0]['type'] == '政治的組織名'

    def test_hospital_corp_to_fac(self):
        recs = JapaneseDatasetLoader._to_records(_rows())
        hosp = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '2942700' and g['text'] == 'セントメアリー病院'
        ]
        assert hosp and hosp[0]['type'] == '施設名'

    def test_university_kept_as_is(self):
        # 대학 법인 본체는 이미 HF 원본이 法人名이므로 LABEL_CORRECTIONS
        # 적용 대상이 아니다. 원본 라벨이 그대로 유지되는지 확인.
        recs = JapaneseDatasetLoader._to_records(_rows())
        univ = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '2942700' and g['text'] == 'コロラド・メサ大学'
        ]
        assert univ and univ[0]['type'] == '法人名'

    def test_unrelated_entities_untouched(self):
        recs = JapaneseDatasetLoader._to_records(_rows())
        other = [g for r in recs for g in r['gold_spans'] if r['id'] == '999999']
        assert other and other[0]['type'] == '人名'

    def test_corrections_table_size(self):
        # 이슈 #16 범위: NHK 1건 + 政府 5건 + 병원 1건 = 7건
        assert len(JapaneseDatasetLoader.LABEL_CORRECTIONS) == 7
