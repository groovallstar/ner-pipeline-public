"""언어 감지 후보 2종 — 손규칙(프로덕션 detect_lang) vs fastText-LID.

손규칙 후보는 프로덕션 `server.detect.detect_lang` 을 그대로 쓴다 — 벤치가
사본이 아니라 실제 출하 코드를 시험하게 해 드리프트를 원천 차단한다. 두
후보 모두 가나 보유 시 `ja` 로 우선 확정(override)하고, 가나가 없을 때만
본 판정으로 vi / unsupported 를 가른다.

- hand_rules: detect_lang — vi-변별 코드포인트(horn·hook·dot + đ)가 있으면
  vi, 아니면 unsupported. 범-라틴 부호는 fr/pt/de/es/tr 와 공유돼 신호가
  아니다.
- fasttext_lid: lid.176 예측이 vi 면 vi, 그 외(en·zh·ja·fr…)는 unsupported.
  미지원을 en 으로 단정하지 않는다 — ja·vi 신호 부재를 감지할 뿐이다.
"""

from typing import Callable

from server.detect import detect_lang, has_kana

# 손규칙 후보 = 프로덕션 감지기(단일 출처)
hand_rules = detect_lang


def make_fasttext_lid(model_path: str) -> Callable[[str], str]:
    """fastText-LID 후보 생성 — lid.176 모델을 1회 로드해 클로저로 반환.

    가나→ja override 는 손규칙과 동일(프로덕션 has_kana 재사용). 가나 부재
    시 LID 예측이 vi 면 vi, 그 외 라벨은 모두 unsupported 로 떨어뜨린다.
    """
    import fasttext  # 후보 벤치 전용 의존 — 런타임 채택 시 pyproject 승격
    model = fasttext.load_model(str(model_path))

    def detect(text: str) -> str:
        if has_kana(text):
            return 'ja'
        # fastText 는 개행을 허용하지 않아 공백으로 치환
        flat = text.replace('\n', ' ').strip()
        if not flat:
            return 'unsupported'
        labels, _ = model.predict(flat)
        code = labels[0].removeprefix('__label__')
        return 'vi' if code == 'vi' else 'unsupported'

    return detect
