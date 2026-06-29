"""ja/vi NER REST API 서버.

학습된 BERT 분류기(`/data/ner/{ja,vi}/model/`)를 감싸 단일·배치
엔드포인트로 char-offset 엔티티 span 을 제공한다. 추론 코어는 안정적
`ner.classifier.data_utils` 만 의존하며, 임계값 로딩·적용은 서버 내부에
자족 구현(`thresholds`)으로 둔다.

기동: `python -m server`
"""
