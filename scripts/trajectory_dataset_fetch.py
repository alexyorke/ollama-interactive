from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from huggingface_hub import HfApi, snapshot_download
except ModuleNotFoundError:
    HfApi = None
    snapshot_download = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import trajectory_dataset_catalog as catalog
from scripts import trajectory_profile


DEFAULT_DATA_ROOT = trajectory_profile.DEFAULT_DATA_ROOT
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "trajectory-dataset-fetch.json"
MANIFEST_NAME = ".ollama-interactive-manifest.json"


def _supported_dataset_specs() -> dict[str, dict[str, Any]]:
    catalog_specs = {
        str(spec.get("slug") or ""): spec
        for spec in catalog.HF_DATASET_CANDIDATES
        if str(spec.get("slug") or "").strip()
    }
    merged: dict[str, dict[str, Any]] = {}
    for dataset_name, profile_spec in trajectory_profile.DATASET_SPECS.items():
        catalog_spec = catalog_specs.get(dataset_name)
        if not catalog_spec:
            continue
        merged[dataset_name] = {
            "slug": dataset_name,
            "repo_id": str(catalog_spec["repo_id"]),
            "adapter": str(profile_spec["adapter"]),
            "path_globs": list(profile_spec["paths"]),
            "priority": str(catalog_spec.get("priority") or ""),
            "reason": str(catalog_spec.get("reason") or ""),
        }
    return merged


SUPPORTED_DATASET_SPECS = _supported_dataset_specs()
DEFAULT_DATASETS = tuple(SUPPORTED_DATASET_SPECS.keys())


def dataset_manifest_path(data_root: Path, dataset: str) -> Path:
    return data_root / dataset / MANIFEST_NAME


def read_dataset_manifest(data_root: Path, dataset: str) -> dict[str, Any] | None:
    path = dataset_manifest_path(data_root, dataset)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _require_hf_download_support() -> tuple[type[Any], Callable[..., str]]:
    if HfApi is None or snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required for trajectory downloads. "
            "Install it with `python -m pip install huggingface_hub`."
        )
    return HfApi, snapshot_download


def _matching_files(dataset_dir: Path, path_globs: list[str]) -> list[str]:
    matched: set[str] = set()
    for pattern in path_globs:
        for path in dataset_dir.glob(pattern):
            if path.is_file():
                matched.add(path.relative_to(dataset_dir).as_posix())
    return sorted(matched)


def _manifest_payload(
    *,
    dataset: str,
    repo_id: str,
    dataset_dir: Path,
    requested_revision: str | None,
    resolved_revision: str | None,
    adapter: str,
    path_globs: list[str],
    files: list[str],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "repo_id": repo_id,
        "adapter": adapter,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(dataset_dir.parent.resolve(strict=False)),
        "local_dir": str(dataset_dir.resolve(strict=False)),
        "path_globs": list(path_globs),
        "files": list(files),
        "file_count": len(files),
    }


def fetch_datasets(
    datasets: list[str],
    data_root: Path,
    *,
    revision: str | None = None,
    api: Any | None = None,
    downloader: Callable[..., str] | None = None,
) -> dict[str, Any]:
    api_class: type[Any] | None = None
    default_downloader: Callable[..., str] | None = None
    if api is None or downloader is None:
        api_class, default_downloader = _require_hf_download_support()
    data_root.mkdir(parents=True, exist_ok=True)
    hf_api = api or api_class()
    download = downloader or default_downloader
    assert download is not None
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        spec = SUPPORTED_DATASET_SPECS[dataset]
        dataset_dir = data_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_manifest_path(data_root, dataset)
        repo_id = str(spec["repo_id"])
        path_globs = list(spec["path_globs"])
        try:
            info = hf_api.dataset_info(repo_id, revision=revision)
            resolved_revision = str(getattr(info, "sha", "") or revision or "")
            download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=path_globs,
                local_dir=str(dataset_dir),
            )
            files = _matching_files(dataset_dir, path_globs)
            manifest = _manifest_payload(
                dataset=dataset,
                repo_id=repo_id,
                dataset_dir=dataset_dir,
                requested_revision=revision,
                resolved_revision=resolved_revision or None,
                adapter=str(spec["adapter"]),
                path_globs=path_globs,
                files=files,
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            rows.append(
                {
                    "dataset": dataset,
                    "repo_id": repo_id,
                    "status": "downloaded" if files else "empty",
                    "ok": bool(files),
                    "resolved_revision": manifest["resolved_revision"],
                    "local_dir": manifest["local_dir"],
                    "manifest_path": str(manifest_path.resolve(strict=False)),
                    "file_count": len(files),
                    "files": files,
                }
            )
        except Exception as exc:
            try:
                manifest_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
            rows.append(
                {
                    "dataset": dataset,
                    "repo_id": repo_id,
                    "status": "error",
                    "ok": False,
                    "error": str(exc),
                    "local_dir": str(dataset_dir.resolve(strict=False)),
                    "manifest_path": str(manifest_path.resolve(strict=False)),
                    "file_count": 0,
                    "files": [],
                }
            )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve(strict=False)),
        "requested_revision": revision,
        "datasets_requested": list(datasets),
        "datasets": rows,
        "summary": {
            "requested": len(datasets),
            "downloaded": len([row for row in rows if row.get("status") == "downloaded"]),
            "empty": len([row for row in rows if row.get("status") == "empty"]),
            "errors": len([row for row in rows if row.get("status") == "error"]),
        },
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download supported public trajectory datasets into the local scratch data root.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(SUPPORTED_DATASET_SPECS),
        default=list(DEFAULT_DATASETS),
        help="Supported public datasets to download.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory for local trajectory datasets.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision to pin for every requested dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON summary output path.")
    args = parser.parse_args(argv)

    try:
        payload = fetch_datasets(list(args.datasets), args.data_root, revision=args.revision)
    except Exception as exc:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_root": str(args.data_root.resolve(strict=False)),
            "requested_revision": args.revision,
            "datasets_requested": list(args.datasets),
            "datasets": [],
            "summary": {"requested": len(args.datasets), "downloaded": 0, "empty": 0, "errors": 1},
            "error": str(exc),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    print(
        "[trajectory-dataset-fetch] "
        + f"requested={summary.get('requested', 0)} "
        + f"downloaded={summary.get('downloaded', 0)} "
        + f"empty={summary.get('empty', 0)} "
        + f"errors={summary.get('errors', 0)}"
    )
    if payload.get("error"):
        print(f"[trajectory-dataset-fetch] error={payload['error']}")
    for row in payload.get("datasets", []):
        if not isinstance(row, dict):
            continue
        print(
            "[trajectory-dataset-fetch] "
            + f"dataset={row.get('dataset')} "
            + f"status={row.get('status')} "
            + f"files={row.get('file_count', 0)} "
            + f"revision={row.get('resolved_revision') or '-'}"
        )
    return 0 if not payload.get("error") and int(summary.get("errors", 0) or 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
