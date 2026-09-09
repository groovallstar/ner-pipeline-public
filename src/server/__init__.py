"""ja/ko/vi/en NER REST API 서버.

학습된 BERT 분류기(`/data/ner/{ja,ko,vi,en}/model/`)를 감싸 단일·배치
엔드포인트로 char-offset 엔티티 span 을 제공한다. 추론 코어는 안정적
`ner.classifier` 의 data_utils(인코딩·디코드)와 confidence_threshold
(임계값 로딩·적용)에만 의존하고, 학습·평가 모듈은 import 하지 않는다.

기동: `python -m server`
"""
