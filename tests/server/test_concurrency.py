"""동시성 guard 계약 테스트 — in-flight 불변식·큐 상한·타임아웃(결정적).

ConcurrencyGuard 를 직접 asyncio 로 구동해 (a) 동시 in-flight ≤ N 항상,
(b) 충분 부하 시 max in-flight = N 도달(직렬화 배제), 그리고 큐 상한·
타임아웃 초과 시 Overloaded(429) 를 단언한다. 실모델·GPU 불요 — 순수
동시성 계약만 본다.
"""

import asyncio

from server.concurrency import ConcurrencyGuard, Overloaded


async def _swallow(guard):
    """큐 대기자 역할 — 슬롯을 얻으면 바로 반납한다."""
    async with guard:
        await asyncio.sleep(0)


def test_in_flight_bounded_and_saturates():
    """동시 in-flight ≤ N 항상 + 충분 부하 시 max in-flight = N 도달."""
    async def scenario():
        guard = ConcurrencyGuard(max_concurrency=3, max_queue=100,
                                 acquire_timeout_s=5.0)
        observed_max = 0
        violations = []

        async def task():
            nonlocal observed_max
            async with guard:
                cur = guard.in_flight
                observed_max = max(observed_max, cur)
                if cur > 3:
                    violations.append(cur)
                await asyncio.sleep(0.02)  # 동시 점유 유지

        await asyncio.gather(*[task() for _ in range(12)])
        return observed_max, violations

    observed_max, violations = asyncio.run(scenario())
    assert violations == []        # (a) 불변식: 항상 ≤ N
    assert observed_max == 3        # (b) 포화: max = N 도달


def test_queue_full_rejects_with_overloaded():
    """대기 큐가 max_queue 를 넘으면 즉시 Overloaded."""
    async def scenario():
        guard = ConcurrencyGuard(max_concurrency=1, max_queue=1,
                                 acquire_timeout_s=5.0)
        release = asyncio.Event()

        async def holder():
            async with guard:
                await release.wait()

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.01)              # holder 가 슬롯 점유
        w = asyncio.create_task(_swallow(guard))  # waiting=1(큐 가득)
        await asyncio.sleep(0.01)
        rejected = False
        try:
            async with guard:                  # waiting(1) >= max_queue(1)
                pass
        except Overloaded:
            rejected = True
        release.set()
        await h
        await w
        return rejected

    assert asyncio.run(scenario()) is True


def test_zero_queue_still_admits_when_slot_free():
    """max_queue=0 이어도 빈 슬롯이 있으면 거절하지 않는다(엣지)."""
    async def scenario():
        guard = ConcurrencyGuard(max_concurrency=2, max_queue=0,
                                 acquire_timeout_s=5.0)
        ok = 0
        async with guard:      # 슬롯 가용 → 통과
            ok += 1
            async with guard:  # 두 번째 슬롯도 가용 → 통과
                ok += 1
        return ok

    assert asyncio.run(scenario()) == 2


def test_acquire_timeout_rejects_with_overloaded():
    """슬롯을 타임아웃 안에 못 얻으면 Overloaded."""
    async def scenario():
        guard = ConcurrencyGuard(max_concurrency=1, max_queue=100,
                                 acquire_timeout_s=0.05)
        release = asyncio.Event()

        async def holder():
            async with guard:
                await release.wait()

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.01)              # 슬롯 점유
        timed_out = False
        try:
            async with guard:                  # 0.05s 대기 후 타임아웃
                pass
        except Overloaded:
            timed_out = True
        release.set()
        await h
        return timed_out

    assert asyncio.run(scenario()) is True
