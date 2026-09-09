# 보관된 실험 결과

제품 문서가 인용하는 결과 JSON을 보관한다. 보관된 JSON의 출처는
`/work/git/ner-pipeline`의 `6c7a64f` 작업 트리이며 JSON은 바이트 그대로 복사했다.
파일 안의 실행 조건과 대응하는 이슈·리포트로 출처를 확인한다.

결과는 실제 실행 산출물에서 복사하고 수치를 손으로 작성하거나 덮어쓰지 않는다.
모델 가중치와 예측 덤프는 포함하지 않는다. 새 인용은 루트 AGENTS.md의
출처·실행 조건 기록 원칙을 따른다.

문서의 `<!-- certified: classifier/ko/<run>/metrics.json -->` 선언은 이
디렉터리 기준 상대경로이며 파일 또는 디렉터리를 가리킨다.
`uv run pytest tests/test_certified_declarations.py -q`는 선언의 존재와 경계만
검사한다. 수치 대조나 재현성을 보증하는 자동 게이트는 없다.
