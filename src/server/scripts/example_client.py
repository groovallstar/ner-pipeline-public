"""내부 소비자용 ja·ko·vi·en NER REST API 호출 예제·자기검증.

내부망 별도 프로세스가 서버를 어떻게 호출하는지 보여주는 최소 레퍼런스다.
`NERClient` 로 단일·배치·자동감지를 호출하고, 데모는 정상 경로와 계약 에러
(400 택일 위반·413 크기 초과)·미지원(unsupported)을 한 번씩 호출해
응답을 출력하고 기대 결과를 검증한다 — 통과 시 0, 하나라도 어긋나면 1 로
종료한다(실서버 대상 스모크 겸 소비자 레퍼런스).
429 부하 검사는 `--overload`를 지정했을 때만 실행한다.

사용:
    uv run python -m server.scripts.example_client \\
        --base-url http://localhost:8008 [--api-key KEY] [--overload]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests


class NERClient:
    """ja·ko·vi·en NER 서버 클라이언트(내부망 별도 프로세스용 최소 래퍼).

    `ner_single`/`ner_batch` 는 `requests.Response` 를 그대로 돌려준다 —
    호출자가 `status_code` 와 `json()` 으로 정상/에러 계약을 직접 판단하게
    해, 413·429 같은 계약 에러를 삼키지 않는다. `max_retries`를 지정하면
    429에 한해서 추가 시도하며, 상한에 도달하면 마지막 응답을 반환한다.
    """

    def __init__(self, base_url: str = 'http://localhost:8008',
                 api_key: Optional[str] = None, timeout: float = 30.0,
                 max_retries: int = 0):
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError('max_retries must be a non-negative integer')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = {'Content-Type': 'application/json'}
        if api_key:
            self._headers['x-api-key'] = api_key

    def health(self) -> dict:
        """서버 상태·언어별 로드 여부(dict)."""
        r = requests.get(f'{self.base_url}/health', timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def ner_single(self, text: str,
                   lang: Optional[str] = None) -> requests.Response:
        """단일 텍스트 NER. `lang` 생략 시 서버가 자동감지한다."""
        body: dict = {'text': text}
        if lang:
            body['lang'] = lang
        return self._post(body)

    def ner_batch(self, texts: List[str],
                  lang: Optional[str] = None) -> requests.Response:
        """배치 텍스트 NER. 응답 `results` 는 입력 순서와 1:1."""
        body: dict = {'texts': texts}
        if lang:
            body['lang'] = lang
        return self._post(body)

    def _post(self, body: dict) -> requests.Response:
        """429만 선택적으로 재시도한다. 현재 서버의 Retry-After는 초 단위다."""
        remaining = self.max_retries
        while True:
            response = requests.post(f'{self.base_url}/v1/ner', json=body,
                                     headers=self._headers, timeout=self.timeout)
            if response.status_code != 429 or remaining == 0:
                return response
            try:
                delay = int(response.headers.get('Retry-After', '1'))
            except ValueError:
                delay = 1
            if delay < 0:
                delay = 1
            response.close()
            time.sleep(delay)
            remaining -= 1


def _check(desc: str, ok: bool, detail: str = '') -> bool:
    """검증 1건 결과를 출력하고 통과 여부를 반환한다."""
    mark = 'ok' if ok else 'FAIL'
    print(f'  {mark}: {desc}{(" -- " + detail) if detail else ""}')
    return ok


def run_demo(client: NERClient, max_chars: int = 20000,
             *, overload: bool = False) -> int:
    """일반 계약을 검사하고 overload 지정 시 부하도 검사한다(0=PASS)."""
    ok = True

    # 정상: 단일 ja 자동감지 → PER/LOC
    r = client.ner_single('織田信長は東京都千代田区に住んでいた。')
    body = r.json()
    ok &= _check(
        'single ja -> 200 + PER/LOC',
        r.status_code == 200 and body.get('lang') == 'ja'
        and any(e['label'] == 'PER' for e in body['entities']), str(body))

    # 정상: 배치 ja+ko+vi+en 자동감지(순서 1:1 보존)
    r = client.ner_batch(['トヨタは日本の会社です。', '삼성전자는 수원에 있다.',
                          'Hà Nội là thủ đô.',
                          'Barack Obama was born in Hawaii.'])
    body = r.json()
    langs = [item['lang'] for item in body.get('results', [])]
    ok &= _check('batch -> ja,ko,vi,en order preserved',
                 r.status_code == 200 and langs == ['ja', 'ko', 'vi', 'en'],
                 str(langs))

    # 미지원: 한자만 → 200 + unsupported(모델 미호출, 빈 결과)
    r = client.ner_single('東京都千代田区')
    body = r.json()
    ok &= _check(
        'unsupported -> 200 + empty',
        r.status_code == 200 and body.get('lang') == 'unsupported'
        and body['entities'] == [], str(body))

    # 에러 400: text·texts 동시 누락(택일 위반)
    r = client._post({})
    ok &= _check('oneof violation -> 400', r.status_code == 400,
                 str(r.status_code))

    # 에러 413: 텍스트 1건 max_chars 초과(크기 가드)
    r = client.ner_single('あ' * (max_chars + 1))
    ok &= _check('text exceeds max_chars -> 413', r.status_code == 413,
                 str(r.status_code))

    if overload:
        # 에러 429: 동시 과부하. 짧은 텍스트를 동시 대량 발사해 in-flight+큐
        # 초과를 유발한다. 기본 설정은 추론 8건 + 대기 32건이다. 서버 설정과
        # 부하 상황에 따라 429가 없을 수 있으므로 실서버 검증 결과를 확인한다.
        def _hit(_: int) -> int:
            try:
                return client.ner_single('東京', lang='ja').status_code
            except requests.RequestException:
                return -1

        with ThreadPoolExecutor(max_workers=64) as ex:
            codes = list(ex.map(_hit, range(200)))
        ok &= _check('concurrent overload -> 429',
                     429 in codes,
                     f'200:{codes.count(200)} 429:{codes.count(429)}')

    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(
        description='Example NER API client + live self-check.')
    p.add_argument('--base-url', default='http://localhost:8008',
                   help='Server base URL')
    p.add_argument('--api-key', default=None,
                   help='x-api-key header (if server auth is enabled)')
    p.add_argument('--max-chars', type=int, default=20000,
                   help='Server max_chars (for the 413 check)')
    p.add_argument('--overload', action='store_true',
                   help='Run the overload check: 200 requests with up to '
                        '64 workers; adds load and requires a 429 response')
    args = p.parse_args()

    client = NERClient(args.base_url, api_key=args.api_key)
    print('health:', client.health())
    code = run_demo(client, max_chars=args.max_chars, overload=args.overload)
    print('DEMO PASS' if code == 0 else 'DEMO FAIL')
    sys.exit(code)


if __name__ == '__main__':
    main()
