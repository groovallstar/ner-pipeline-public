"""CLI 진입점: python -m ner.llm_eval.benchmark

사용 예:
    python -m ner.llm_eval.benchmark \
        --models vllm:Qwen/Qwen3.5-9B hf:model-name openai:gpt-4o-mini \
        --max-samples 100 \
        --output results/benchmark.json
"""

import argparse
import logging
import os
import sys

from ner.labelers.dataset_loader import DatasetLoader  # noqa: direct import avoids bs4 dep in labelers.__init__
from ner.llm_eval.benchmark_runner import BenchmarkRunner
from ner.llm_eval.report import ReportGenerator


def _load_env():
    """python-dotenv가 설치된 경우 .env 파일을 로드하고, 없으면 건너뛴다."""
    env_path = os.path.join(os.path.dirname(__file__), "../../../.env")
    env_path = os.path.abspath(env_path)
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            # 수동 폴백: KEY=VALUE 형식으로 파싱한다
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value


def _create_labeler(model_spec: str, args, lang: str = "ko"):
    """모델 스펙을 파싱하여 적절한 라벨러를 생성한다.

    형식: backend:model_name
    예시:
        vllm:Qwen/Qwen3.5-9B
        openai:gpt-5-mini
        hf:soddokayo/klue-roberta-large-klue-ner
    """
    parts = model_spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid model spec '{model_spec}'. Use format 'backend:model_name'")

    backend, model_name = parts

    if lang == "ja":
        return _create_labeler_ja(backend, model_name, args)
    if lang == "vi":
        return _create_labeler_vi(backend, model_name, args)

    if backend == "vllm":
        from ner.labelers.ko.vllm_ner_labeler import VllmNERLabeler
        return backend, VllmNERLabeler(
            base_url=args.vllm_url,
            model=model_name,
            concurrency=args.concurrency,
            thinking=getattr(args, "thinking", False),
        )
    elif backend == "openai":
        from ner.labelers.ko.openai_ner_labeler import OpenAINERLabeler
        return backend, OpenAINERLabeler(model=model_name)
    elif backend == "hf":
        from ner.labelers.hf_ner_labeler import HFNERLabeler
        return backend, HFNERLabeler(model_name=model_name, lang="ko")
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use: vllm, openai, hf")


def _create_labeler_ja(backend: str, model_name: str, args):
    """일본어 NER 라벨러를 생성한다."""
    if backend == "vllm":
        from ner.labelers.ja.vllm_ner_labeler import VllmNERLabeler
        return backend, VllmNERLabeler(
            base_url=args.vllm_url,
            model=model_name,
            concurrency=args.concurrency,
            thinking=getattr(args, "thinking", False),
        )
    elif backend == "openai":
        from ner.labelers.ja.openai_ner_labeler import OpenAINERLabeler
        return backend, OpenAINERLabeler(model=model_name)
    else:
        raise ValueError(f"Unknown backend '{backend}' for Japanese. Use: vllm, openai")


def _create_labeler_vi(backend: str, model_name: str, args):
    """베트남어 NER 라벨러를 생성한다."""
    if backend == "vllm":
        from ner.labelers.vi.vllm_ner_labeler import VllmNERLabeler
        return backend, VllmNERLabeler(
            base_url=args.vllm_url,
            model=model_name,
            concurrency=args.concurrency,
            thinking=getattr(args, "thinking", False),
        )
    elif backend == "openai":
        from ner.labelers.vi.openai_ner_labeler import OpenAINERLabeler
        return backend, OpenAINERLabeler(model=model_name)
    elif backend == "hf":
        from ner.labelers.hf_ner_labeler import HFNERLabeler
        return backend, HFNERLabeler(model_name=model_name, lang="vi")
    else:
        raise ValueError(f"Unknown backend '{backend}' for Vietnamese. Use: vllm, openai, hf")


def main():
    _load_env()

    parser = argparse.ArgumentParser(
        description="NER Benchmark: evaluate labeling quality and speed",
        prog="python -m ner.llm_eval.benchmark",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model specs: 'backend:model' (e.g. vllm:Qwen/Qwen3.5-9B, openai:gpt-5-mini)",
    )
    parser.add_argument("--lang", default="ko", choices=["ko", "ja", "vi"], help="Language (default: ko)")
    parser.add_argument("--dataset", default=None, help="Dataset name (default: klue for ko, stockmark for ja)")
    parser.add_argument("--local-file", default=None, help="Local JSONL gold file (e.g. PII-injected dataset); ja, vi only")
    parser.add_argument("--config", default="ner", help="Dataset config (default: ner)")
    parser.add_argument("--split", default=None, help="Dataset split (default: validation for ko, test for ja)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output", default=None, help="JSON output path")

    # Backend-specific options
    parser.add_argument("--vllm-url", default="http://localhost:8081/v1", help="vLLM base URL")
    parser.add_argument("--concurrency", type=int, default=32, help="vLLM concurrency")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode for vLLM")

    args = parser.parse_args()

    # Apply language-specific defaults
    if args.dataset is None:
        if args.lang == "ja":
            args.dataset = "stockmark/ner-wikipedia-dataset"
        elif args.lang == "vi":
            args.dataset = "unimelb-nlp/wikiann"
        else:
            args.dataset = "klue"
    if args.split is None:
        args.split = "test" if args.lang in ("ja", "vi") else "validation"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _run_benchmark(args)


def _load_gold(args):
    """언어별 gold 레코드를 로드한다."""
    if args.lang == "ja":
        from ner.labelers.ja.dataset_loader import JapaneseDatasetLoader
        if args.local_file:
            print(f"Loading gold data (local): {args.local_file}")
            return JapaneseDatasetLoader.load_local(
                args.local_file, max_samples=args.max_samples,
            )
        loader = JapaneseDatasetLoader()
        path = loader.DEFAULT_PATH[args.split]
        print(f"Loading gold data (canonical dump): {path}")
        return loader.load(split=args.split, max_samples=args.max_samples)
    if args.lang == "vi":
        from ner.labelers.vi.dataset_loader import VietnameseDatasetLoader
        if args.local_file:
            print(f"Loading gold data (local): {args.local_file}")
            return VietnameseDatasetLoader.load_local(
                args.local_file, max_samples=args.max_samples,
            )
        loader = VietnameseDatasetLoader()
        path = loader.DEFAULT_PATH[args.split]
        print(f"Loading gold data (canonical dump): {path}")
        return loader.load(split=args.split, max_samples=args.max_samples)
    print(f"Loading gold data: {args.dataset}/{args.config} [{args.split}]")
    return DatasetLoader().load(
        args.dataset, config=args.config, split=args.split, max_samples=args.max_samples,
    )


def _eval_mode_for_lang(lang: str) -> str:
    """언어별 평가 모드를 반환한다.

    - ja, vi: 문자 오프셋 span (gold_spans 스키마)
    - ko: 음절 BIO (tokens + ner_tags 스키마)
    """
    if lang in ("ja", "vi"):
        return "offset_span"
    return "bio"


def _run_benchmark(args):
    """통합 벤치마크 러너: eval_mode를 통해 언어별로 분기한다."""
    gold_records = _load_gold(args)
    print(f"  Loaded {len(gold_records)} records")

    eval_mode = _eval_mode_for_lang(args.lang)
    runner = BenchmarkRunner(
        gold_records,
        lang=args.lang,
        eval_mode=eval_mode,
        sample_concurrency=args.concurrency,
    )

    for spec in args.models:
        try:
            backend, labeler = _create_labeler(spec, args, lang=args.lang)
            display_name = spec.split(":", 1)[1] if ":" in spec else spec
            runner.add_labeler(display_name, backend, labeler)
            print(f"  Added: [{backend}] {display_name}")
        except Exception as e:
            print(f"  ERROR creating labeler for '{spec}': {e}", file=sys.stderr)
            continue

    results = runner.run()
    report = ReportGenerator(results, lang=args.lang)
    report.print_table()
    if args.output:
        report.save_json(args.output)


if __name__ == "__main__":
    main()
