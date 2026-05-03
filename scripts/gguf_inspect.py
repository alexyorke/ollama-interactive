from __future__ import annotations

import argparse
import json
import math
import os
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GGUF_VALUE_TYPES = {
    0: ("uint8", 1, "B"),
    1: ("int8", 1, "b"),
    2: ("uint16", 2, "H"),
    3: ("int16", 2, "h"),
    4: ("uint32", 4, "I"),
    5: ("int32", 4, "i"),
    6: ("float32", 4, "f"),
    7: ("bool", 1, "?"),
    8: ("string", None, None),
    9: ("array", None, None),
    10: ("uint64", 8, "Q"),
    11: ("int64", 8, "q"),
    12: ("float64", 8, "d"),
}

GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "TQ1_0",
    32: "TQ2_0",
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: list[int]
    type_id: int
    offset: int

    @property
    def ggml_type(self) -> str:
        return GGML_TYPES.get(self.type_id, f"type_{self.type_id}")

    @property
    def params(self) -> int:
        return math.prod(self.dims) if self.dims else 0


@dataclass(frozen=True)
class ManifestModel:
    tag: str
    manifest: Path
    blob: Path
    size: int


class Reader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("rb")

    def close(self) -> None:
        self.handle.close()

    def read_exact(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise ValueError(f"unexpected EOF while reading {self.path}")
        return data

    def unpack(self, fmt: str) -> Any:
        return struct.unpack("<" + fmt, self.read_exact(struct.calcsize("<" + fmt)))[0]

    def string(self) -> str:
        size = self.unpack("Q")
        return self.read_exact(size).decode("utf-8", errors="replace")

    def skip_value(self, type_id: int) -> None:
        name, size, _fmt = GGUF_VALUE_TYPES[type_id]
        if name == "string":
            size = self.unpack("Q")
            self.handle.seek(size, os.SEEK_CUR)
            return
        if name == "array":
            elem_type = self.unpack("I")
            count = self.unpack("Q")
            self.skip_array(elem_type, count)
            return
        if size is None:
            raise ValueError(f"cannot skip GGUF type {type_id}")
        self.handle.seek(size, os.SEEK_CUR)

    def skip_array(self, elem_type: int, count: int) -> None:
        name, size, _fmt = GGUF_VALUE_TYPES[elem_type]
        if name == "string":
            for _ in range(count):
                item_size = self.unpack("Q")
                self.handle.seek(item_size, os.SEEK_CUR)
            return
        if name == "array":
            for _ in range(count):
                nested_type = self.unpack("I")
                nested_count = self.unpack("Q")
                self.skip_array(nested_type, nested_count)
            return
        if size is None:
            raise ValueError(f"cannot skip GGUF array type {elem_type}")
        self.handle.seek(size * count, os.SEEK_CUR)

    def metadata_value(self, type_id: int) -> Any:
        name, size, fmt = GGUF_VALUE_TYPES[type_id]
        if name == "string":
            return self.string()
        if name == "array":
            elem_type = self.unpack("I")
            count = self.unpack("Q")
            elem_name = GGUF_VALUE_TYPES.get(elem_type, (f"type_{elem_type}", None, None))[0]
            self.skip_array(elem_type, count)
            return {"type": f"array<{elem_name}>", "len": count}
        if size is None or fmt is None:
            raise ValueError(f"unsupported GGUF type {type_id}")
        return self.unpack(fmt)


def ollama_models_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")


def digest_to_blob(models_dir: Path, digest: str) -> Path:
    return models_dir / "blobs" / digest.replace(":", "-")


def manifest_tag(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).parts)
    if len(parts) < 2:
        return path.name
    tag = parts[-1]
    body = parts[:-1]
    if body[:2] == ["registry.ollama.ai", "library"]:
        name = "/".join(body[2:])
    elif body and body[0] == "registry.ollama.ai":
        name = "/".join(body[1:])
    else:
        name = "/".join(body)
    return f"{name}:{tag}"


def load_manifests(models_dir: Path) -> dict[str, ManifestModel]:
    root = models_dir / "manifests"
    found: dict[str, ManifestModel] = {}
    if not root.exists():
        return found
    for manifest in root.rglob("*"):
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for layer in payload.get("layers") or []:
            media_type = str(layer.get("mediaType") or "")
            if not media_type.endswith(".model"):
                continue
            digest = str(layer.get("digest") or "")
            if not digest.startswith("sha256:"):
                continue
            tag = manifest_tag(manifest, root)
            found[tag] = ManifestModel(
                tag=tag,
                manifest=manifest,
                blob=digest_to_blob(models_dir, digest),
                size=int(layer.get("size") or 0),
            )
    return found


def inspect_gguf(path: Path, *, include_metadata: bool = False) -> dict[str, Any]:
    reader = Reader(path)
    try:
        magic = reader.read_exact(4)
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = reader.unpack("I")
        tensor_count = reader.unpack("Q")
        metadata_count = reader.unpack("Q")

        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.string()
            type_id = reader.unpack("I")
            if type_id not in GGUF_VALUE_TYPES:
                raise ValueError(f"unknown GGUF metadata type {type_id} for {key}")
            metadata[key] = reader.metadata_value(type_id)

        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.unpack("I")
            dims = [reader.unpack("Q") for _ in range(n_dims)]
            type_id = reader.unpack("I")
            offset = reader.unpack("Q")
            tensors.append(TensorInfo(name=name, dims=dims, type_id=type_id, offset=offset))
    finally:
        reader.close()

    arch = str(metadata.get("general.architecture") or "")
    prefix = f"{arch}." if arch else ""
    vocab_info = metadata.get("tokenizer.ggml.tokens")
    vocab_size = vocab_info.get("len") if isinstance(vocab_info, dict) else None
    hidden = metadata.get(f"{prefix}embedding_length")
    context = metadata.get(f"{prefix}context_length")
    blocks = metadata.get(f"{prefix}block_count")
    ff = metadata.get(f"{prefix}feed_forward_length")
    heads = metadata.get(f"{prefix}attention.head_count")
    kv_heads = metadata.get(f"{prefix}attention.head_count_kv")

    by_name = {tensor.name: tensor for tensor in tensors}
    output_tensor = by_name.get("output.weight")
    embed_tensor = by_name.get("token_embd.weight")
    lm_head = output_tensor or embed_tensor
    total_params = sum(tensor.params for tensor in tensors)

    summary: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "gguf_version": version,
        "architecture": arch,
        "file_type": metadata.get("general.file_type"),
        "quantization_version": metadata.get("general.quantization_version"),
        "tokenizer": metadata.get("tokenizer.ggml.model"),
        "vocab_size": vocab_size,
        "context_length": context,
        "embedding_length": hidden,
        "block_count": blocks,
        "feed_forward_length": ff,
        "head_count": heads,
        "kv_head_count": kv_heads,
        "tensor_count": tensor_count,
        "approx_params": total_params,
        "tensor_type_counts": dict(Counter(tensor.ggml_type for tensor in tensors)),
        "output_tensor": tensor_to_dict(output_tensor),
        "embedding_tensor": tensor_to_dict(embed_tensor),
        "lm_head_source": "output.weight" if output_tensor else "token_embd.weight/tied" if embed_tensor else None,
        "lm_head_params": lm_head.params if lm_head else None,
        "lm_head_param_ratio": round(lm_head.params / total_params, 4) if lm_head and total_params else None,
    }
    if include_metadata:
        summary["metadata"] = metadata
    return summary


def tensor_to_dict(tensor: TensorInfo | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "name": tensor.name,
        "dims": tensor.dims,
        "type": tensor.ggml_type,
        "params": tensor.params,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "model",
        "gb",
        "arch",
        "quant",
        "vocab",
        "hidden",
        "ctx",
        "params_b",
        "lm_head_m",
        "lm_%",
        "types",
    ]
    table: list[list[str]] = []
    for row in rows:
        info = row["gguf"]
        type_counts = info.get("tensor_type_counts") or {}
        types = ",".join(f"{name}:{count}" for name, count in sorted(type_counts.items()))
        table.append(
            [
                str(row.get("model") or Path(info["path"]).name[:12]),
                f"{int(info['size_bytes']) / 1e9:.2f}",
                str(info.get("architecture") or ""),
                str(info.get("file_type") or ""),
                str(info.get("vocab_size") or ""),
                str(info.get("embedding_length") or ""),
                str(info.get("context_length") or ""),
                f"{int(info.get('approx_params') or 0) / 1e9:.2f}",
                f"{int(info.get('lm_head_params') or 0) / 1e6:.0f}",
                f"{float(info.get('lm_head_param_ratio') or 0) * 100:.1f}",
                types[:42],
            ]
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in table)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local Ollama GGUF model metadata and output-head cost.")
    parser.add_argument("models", nargs="*", help="Ollama tags, e.g. gemma3:4b qwen3:8b, or GGUF blob paths.")
    parser.add_argument("--models-dir", type=Path, default=ollama_models_dir())
    parser.add_argument("--json", action="store_true", help="Emit full JSON instead of a compact table.")
    parser.add_argument("--include-metadata", action="store_true", help="Include all GGUF metadata values in JSON output.")
    parser.add_argument("--output", type=Path, help="Write JSON payload to this path.")
    args = parser.parse_args(argv)

    manifests = load_manifests(args.models_dir)
    targets = args.models or sorted(manifests)
    rows: list[dict[str, Any]] = []
    for target in targets:
        model = manifests.get(target)
        if model:
            path = model.blob
            label = model.tag
            manifest = str(model.manifest)
        else:
            path = Path(target)
            label = None
            manifest = None
        if not path.exists():
            raise SystemExit(f"model/blob not found: {target}")
        rows.append(
            {
                "model": label,
                "manifest": manifest,
                "gguf": inspect_gguf(path, include_metadata=args.include_metadata),
            }
        )

    payload = {"models_dir": str(args.models_dir), "results": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json or args.output:
        print(json.dumps(payload, indent=2))
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
