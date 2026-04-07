"""Evaluators package.

Imports are lazy to avoid forcing seqeval dependency on Japanese-only paths.
Use direct module imports: e.g. `from evaluators.benchmark_runner import BenchmarkRunner`
"""


def __getattr__(name):
    """Lazy imports for backward compatibility."""
    if name == "TagAligner":
        from evaluators.tag_aligner import TagAligner
        return TagAligner
    if name == "extract_spans_from_bio":
        from evaluators.tag_aligner import extract_spans_from_bio
        return extract_spans_from_bio
    if name == "MetricsCalculator":
        from evaluators.metrics import MetricsCalculator
        return MetricsCalculator
    if name == "BenchmarkRunner":
        from evaluators.benchmark_runner import BenchmarkRunner
        return BenchmarkRunner
    if name == "BenchmarkResult":
        from evaluators.benchmark_runner import BenchmarkResult
        return BenchmarkResult
    if name == "ReportGenerator":
        from evaluators.report import ReportGenerator
        return ReportGenerator
    raise AttributeError(f"module 'evaluators' has no attribute {name!r}")


__all__ = [
    "TagAligner",
    "MetricsCalculator",
    "BenchmarkRunner",
    "BenchmarkResult",
    "ReportGenerator",
]
