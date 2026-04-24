"""Stockmark JA loader — canonical 5종 매핑 및 LABEL_CORRECTIONS 테스트."""
from labelers.ja.dataset_loader import (
    JA_TO_CANONICAL,
    JapaneseDatasetLoader,
)


class _MockRow(dict):
    """HF 데이터셋 row 모사 (dict 접근 + .get)."""


def _rows():
    """canonical 매핑 및 LABEL_CORRECTIONS 대상 사례를 포함한 고정 시퀀스."""
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


class TestCanonicalMapping:
    """HF 일본어 라벨 → canonical 5종 매핑 정합성."""

    def test_mapping_covers_all_ja_labels(self):
        """매핑 테이블이 Stockmark 8종 JA 라벨 모두를 포함한다."""
        assert set(JA_TO_CANONICAL.keys()) == {
            '人名', '法人名', '地名', '施設名',
            '製品名', 'イベント名', '政治的組織名', 'その他の組織名',
        }

    def test_mapping_values_are_5type(self):
        """매핑 결과 라벨이 canonical 5종을 벗어나지 않는다."""
        assert set(JA_TO_CANONICAL.values()) == {
            'PER', 'LOC', 'ORG', 'PROD', 'EVT',
        }

    def test_person_maps_to_per(self):
        assert JA_TO_CANONICAL['人名'] == 'PER'

    def test_location_and_facility_merge_to_loc(self):
        assert JA_TO_CANONICAL['地名'] == 'LOC'
        assert JA_TO_CANONICAL['施設名'] == 'LOC'

    def test_org_family_merge_to_org(self):
        assert JA_TO_CANONICAL['法人名'] == 'ORG'
        assert JA_TO_CANONICAL['政治的組織名'] == 'ORG'
        assert JA_TO_CANONICAL['その他の組織名'] == 'ORG'


class TestToRecordsOutput:
    """_to_records 출력 검증."""

    def test_nhk_becomes_org(self):
        recs = JapaneseDatasetLoader._to_records(_rows())
        nhk = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '2746825' and g['text'] == 'NHK'
        ]
        assert nhk and nhk[0]['type'] == 'ORG'

    def test_government_maps_to_org(self):
        recs = JapaneseDatasetLoader._to_records(_rows())
        gov = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '2919391' and g['text'] == '香港政府'
        ]
        # 정정 없이 法人名 → ORG 매핑만으로 최종 라벨 결정
        assert gov and gov[0]['type'] == 'ORG'

    def test_hospital_correction_lifts_to_loc(self):
        """병원은 LABEL_CORRECTIONS에 남아있어 ORG가 아니라 LOC로 정정된다."""
        recs = JapaneseDatasetLoader._to_records(_rows())
        hosp = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '2942700' and g['text'] == 'セントメアリー病院'
        ]
        assert hosp and hosp[0]['type'] == 'LOC'

    def test_university_maps_to_org(self):
        recs = JapaneseDatasetLoader._to_records(_rows())
        univ = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '2942700' and g['text'] == 'コロラド・メサ大学'
        ]
        assert univ and univ[0]['type'] == 'ORG'

    def test_facility_maps_to_loc(self):
        recs = JapaneseDatasetLoader._to_records(_rows())
        fac = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '2746825' and g['text'] == '仙台放送局'
        ]
        # 施設名 → LOC
        assert fac and fac[0]['type'] == 'LOC'

    def test_person_maps_to_per(self):
        recs = JapaneseDatasetLoader._to_records(_rows())
        per = [
            g for r in recs for g in r['gold_spans']
            if r['id'] == '999999'
        ]
        assert per and per[0]['type'] == 'PER'

    def test_corrections_table_reduced(self):
        """5종 축소 후 실효 정정은 병원 1건만 유지."""
        assert len(JapaneseDatasetLoader.LABEL_CORRECTIONS) == 1
