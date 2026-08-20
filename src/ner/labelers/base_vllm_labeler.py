"""vLLM OpenAI 호환 NER 라벨러의 추상 베이스 클래스.

서브클래스는 언어 팩(entity_types, single_prompt_template, lang)을 주입한다.
"""
import asyncio
import logging
from typing import List, Optional

from openai import AsyncOpenAI

from ner.labelers.llm_helpers import parse_spans, spans_to_bio, split_sentences

logger = logging.getLogger(__name__)


class BaseVllmLabeler:
    """vLLM 컨테이너의 OpenAI 호환 API를 사용하는 배치 NER 라벨러.

    Args:
        base_url: vLLM 서버 URL (예: "http://localhost:8081/v1").
        model: vLLM 컨테이너에서 서빙하는 모델 이름.
        entity_types: NER 태그셋. 기본값은 언어 팩의 기본값을 사용한다.
        max_tokens: 샘플당 최대 생성 토큰 수. 실제로 쓴 양이 아니라 서버가 미리
            잡아두는 자리라, 프롬프트 + 이 값이 max-model-len 을 넘으면 요청이
            GPU 에 닿기도 전에 거절된다. NER 출력은 JSON 배열 한 줄이라 실측
            77 토큰이었고, 1024 는 그 13배 여유다.
        concurrency: vLLM 서버에 대한 최대 동시 요청 수.
        thinking: vLLM 채팅 템플릿의 `enable_thinking` 플래그 활성화 여부.
        lang: 언어 코드("ko"/"ja") — 문장 분리기에 전달된다.
        single_prompt_template: 프롬프트 템플릿 ({entity_types}, {sentence} 포함 f-string).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        model: str = "Qwen/Qwen3.5-27B",
        entity_types: Optional[List[str]] = None,
        max_tokens: int = 1024,
        concurrency: int = 32,
        thinking: bool = False,
        *,
        lang: str = "ko",
        single_prompt_template: str = "",
    ) -> None:
        if not single_prompt_template:
            raise ValueError("single_prompt_template is required")
        self.model = model
        self.entity_types = entity_types or []
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.lang = lang
        self._single_prompt_template = single_prompt_template
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(base_url=base_url, api_key="none")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # 토큰 사용량 추적 (호출 누적, 소비자가 초기화한다)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        logger.info("%s ready: %s @ %s", type(self).__name__, model, base_url)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def _run(self, coro):
        """라벨러 전용 이벤트 루프에서 코루틴을 돌린다.

        `asyncio.run` 은 호출마다 루프를 새로 만들고 닫는데, 클라이언트의 연결 풀과
        세마포어는 **처음 쓴 루프에 묶인다**. 라벨러 하나로 동기 API 를 여러 번 부르면
        (벤치마크가 샘플을 순차 처리하는 방식) 두 번째 호출부터 닫힌 루프의 연결을 잡아
        `Event loop is closed` 가 나고, SDK 재시도가 대개 가려주지만 재시도가 소진되면
        그 샘플이 통째로 실패한다. 루프를 라벨러 수명 동안 하나로 유지하면 그 어긋남이
        생기지 않는다.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def close(self) -> None:
        """전용 루프를 닫는다. 라벨러를 오래 살려두는 호출자를 위한 정리 훅이다."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None

    def label(self, text: str) -> List[dict]:
        """텍스트를 라벨링한다. 문장당 하나의 NER 레코드 리스트를 반환한다."""
        sentences = split_sentences(text, lang=self.lang)
        return self._run(self._label_sentences(sentences, id_offset=0))

    def label_spans(self, text: str, split: bool = True) -> List[dict]:
        """BIO 변환 없이 원시 엔티티 span을 반환한다 (동기 래퍼).

        Args:
            text: 입력 텍스트.
            split: True이면 문장 분리 후 각 문장을 독립 LLM 호출로 처리
                (벤치마크 기본). False이면 전체 텍스트를 단일 프롬프트로
                전달하여 문맥을 유지 (검증 용도 권장).
        """
        return self._run(self.alabel_spans(text, split=split))

    async def alabel_spans(self, text: str, split: bool = True) -> List[dict]:
        """label_spans의 async 버전. 외부 event loop 내에서 호출 가능."""
        if split:
            sentences = split_sentences(text, lang=self.lang)
            non_empty = [s for s in sentences if s.strip()]
        else:
            non_empty = [text] if text.strip() else []

        tasks = [self._chat(s) for s in non_empty]
        raw_results = await asyncio.gather(*tasks)
        all_spans = []
        for raw in raw_results:
            all_spans.extend(parse_spans(raw))
        return all_spans

    def label_records(self, records: List[dict]) -> List[dict]:
        """원시 텍스트 레코드 리스트를 병렬로 라벨링한다.

        입력 레코드는 ``"text"`` 필드를 가져야 한다.
        ``"tokens"``, ``"ner_tags"``, ``"id"`` 필드를 포함한 NER 레코드를 반환한다.
        """
        all_sentences: List[str] = []
        meta: List[tuple] = []

        for r_idx, record in enumerate(records):
            text = record.get("text", "")
            if not text:
                continue
            for s_idx, sentence in enumerate(split_sentences(text, lang=self.lang)):
                all_sentences.append(sentence)
                meta.append((r_idx, s_idx))

        if not all_sentences:
            return []

        return self._run(self._run_batch(all_sentences, meta))

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _build_user_content(self, sentence: str) -> str:
        return self._single_prompt_template.format(
            entity_types=", ".join(self.entity_types),
            sentence=sentence,
        )

    async def _chat(self, sentence: str) -> str:
        """동시성 제한을 적용하여 단일 채팅 완성 요청을 전송한다."""
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self._build_user_content(sentence)}],
                max_tokens=self.max_tokens,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
            )
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens or 0
            self.total_completion_tokens += response.usage.completion_tokens or 0
        return response.choices[0].message.content or ""

    async def _label_sentences(self, sentences: List[str], id_offset: int = 0) -> List[dict]:
        sentences = [s for s in sentences if s.split()]
        if not sentences:
            return []

        raw_outputs = await asyncio.gather(*[self._chat(s) for s in sentences])

        results = []
        for idx, (sentence, raw) in enumerate(zip(sentences, raw_outputs)):
            tokens = sentence.split()
            spans = parse_spans(raw.strip())
            tags = spans_to_bio(tokens, spans)
            results.append({
                "tokens": tokens,
                "ner_tags": tags,
                "id": str(id_offset + idx),
            })
        return results

    async def _run_batch(self, sentences: List[str], meta: List[tuple]) -> List[dict]:
        raw_outputs = await asyncio.gather(*[self._chat(s) for s in sentences])

        results = []
        for (r_idx, s_idx), sentence, raw in zip(meta, sentences, raw_outputs):
            tokens = sentence.split()
            if not tokens:
                continue
            spans = parse_spans(raw.strip())
            tags = spans_to_bio(tokens, spans)
            results.append({
                "tokens": tokens,
                "ner_tags": tags,
                "id": f"{r_idx}-{s_idx}",
            })
        return results
