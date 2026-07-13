"""variance 게이트 세 축(comparability·leakage·noise)이 공유하는 IO·추출 헬퍼.

fold*/metrics.json·pooled_metrics.json 을 읽고 per-entity·overall F1 을
뽑는 순수 함수 모음. 축 모듈이 모두 이 하나를 import 해 중복을 막는다.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: Path) -> dict:
    """JSON 파일을 읽어 dict 로 반환한다."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fold_dirs(run_dir: Path) -> List[Path]:
    """run_dir 아래 fold{N}/metrics.json 을 가진 fold 디렉토리를 fold 번호
    순으로 반환한다."""
    indexed = []
    for p in run_dir.glob("fold*/metrics.json"):
        try:
            idx = int(p.parent.name[len("fold"):])
        except ValueError:
            continue
        indexed.append((idx, p.parent))
    return [d for _, d in sorted(indexed)]


def load_fold_metrics(run_dir: Path) -> List[dict]:
    """run_dir 의 모든 fold metrics.json 을 fold 순서로 로드한다."""
    return [_load_json(d / "metrics.json") for d in _fold_dirs(Path(run_dir))]


def load_pooled_metrics(run_dir: Path) -> dict:
    """run_dir 의 pooled_metrics.json 을 로드한다."""
    return _load_json(Path(run_dir) / "pooled_metrics.json")


def _fold_per_entity(fold: dict, matching: str) -> Dict[str, float]:
    """fold metrics.json 에서 per-entity F1 을 추출한다."""
    per = fold.get(f"per_entity_{matching}") or fold.get("per_entity", {})
    return {e: v["f1"] for e, v in per.items()}


def _fold_overall(fold: dict, matching: str) -> Optional[float]:
    """fold metrics.json 에서 overall F1 을 추출한다."""
    ov = fold.get(f"overall_{matching}") or fold.get("overall")
    return ov["f1"] if ov else None


def _pooled_per_entity(pooled: dict, matching: str) -> Dict[str, float]:
    """pooled_metrics.json 에서 per-entity F1 을 추출한다."""
    per = pooled.get(matching, {}).get("per_entity", {})
    return {e: v["f1"] for e, v in per.items()}


def _pooled_overall(pooled: dict, matching: str) -> Optional[float]:
    """pooled_metrics.json 에서 overall F1 을 추출한다."""
    ov = pooled.get(matching, {}).get("overall")
    return ov["f1"] if ov else None
