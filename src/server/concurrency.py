"""전역 동시성 제어 — 단일 GPU 과부하 방지(in-flight 상한·큐·타임아웃).

추론은 GPU 직렬에 가까워 동시 요청이 무제한이면 지연 절벽·OOM 위험이 있다.
ConcurrencyGuard 는 동시 in-flight 추론을 max_concurrency 로 묶고(세마포어),
대기 큐를 max_queue 로, 대기 시간을 acquire_timeout_s 로 bound 해 무한
대기열을 막는다. 큐/타임아웃 초과는 Overloaded(429)로 빠르게 거절한다.

asyncio 단일 이벤트 루프에서 동작하므로 카운터 증감은 await 경계 밖에서만
일어나 race 가 없다 — 추론 자체는 run_in_threadpool 로 워커에 떠넘기고,
세마포어 점유 구간만 이 guard 가 관장한다.
"""

import asyncio


class Overloaded(Exception):
    """동시성 큐/타임아웃 초과 — API 는 429 로 매핑."""


class ConcurrencyGuard:
    """async 컨텍스트 매니저 — in-flight ≤ max_concurrency 불변식 보장.

    `async with guard:` 진입 시 세마포어를 얻고(필요 시 대기), 나갈 때 푼다.
    대기 중 요청이 max_queue 를 넘으면 즉시 Overloaded, 세마포어를
    acquire_timeout_s 안에 못 얻어도 Overloaded.
    """

    def __init__(self, max_concurrency: int, max_queue: int,
                 acquire_timeout_s: float):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency
        self._max_queue = max_queue
        self._timeout = acquire_timeout_s
        self._waiting = 0
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        """현재 세마포어를 점유한(추론 중) 요청 수."""
        return self._in_flight

    @property
    def waiting(self) -> int:
        """세마포어를 얻으려 대기 중인 요청 수."""
        return self._waiting

    async def __aenter__(self) -> 'ConcurrencyGuard':
        # 슬롯이 없어 대기해야 하는데 큐가 가득이면 거절한다. 빈 슬롯이
        # 있으면 큐 상한과 무관하게 즉시 진입한다 — max_queue=0 이어도 가용
        # 슬롯을 막지 않는다(0 = "대기 불허", "처리 불허"가 아니다).
        if self._sem.locked() and self._waiting >= self._max_queue:
            raise Overloaded(f'queue full (>= {self._max_queue} waiting)')
        self._waiting += 1
        try:
            await asyncio.wait_for(self._sem.acquire(), self._timeout)
        except asyncio.TimeoutError:
            raise Overloaded(f'acquire timed out ({self._timeout}s)')
        finally:
            self._waiting -= 1
        self._in_flight += 1
        return self

    async def __aexit__(self, *exc) -> bool:
        self._in_flight -= 1
        self._sem.release()
        return False
