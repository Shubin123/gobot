#!/usr/bin/env python3
"""
Export the GoBot PyTorch model to ONNX format for browser inference.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model checkpoints/demo_model.pt --num-shards 3
    python scripts/export_onnx.py --output-dir docs/model --opset 17

The script:
1. Loads the best available .pt checkpoint
2. Exports to ONNX (with dynamic batch axis)
3. Optionally shards the ONNX file into N binary chunks
4. Writes a manifest.json describing shards + metadata
5. Prints a summary

WebGPU / ONNX Runtime Web compatibility:
- opset 17 is supported by ort-web 1.18+
- Dynamic batch axis allows batch size 1 in browser
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import hashlib
import struct
import math

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from gobot_engine.neural_net import GoResNet


DEFAULT_CANDIDATES = [
    "checkpoints/winning_gobot_model.pt",
    "checkpoints/demo_model.pt",
    "checkpoints/gobot_model.pt",
]


def find_model(model_path: str | None) -> str:
    if model_path and os.path.exists(model_path):
        return model_path
    for candidate in DEFAULT_CANDIDATES:
        if os.path.exists(candidate):
            print(f"Auto-selected model: {candidate}")
            return candidate
    raise FileNotFoundError(
        f"No model found. Tried: {DEFAULT_CANDIDATES}. "
        "Pass --model <path> to specify one."
    )


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_to_onnx(
    model: GoResNet,
    output_path: str,
    opset: int = 17,
) -> None:
    """Export GoResNet to ONNX with dynamic batch axis."""
    model.eval()
    board_size = model.board_size
    in_channels = model.in_channels

    dummy_input = torch.randn(1, in_channels, board_size, board_size)

    print(f"Exporting ONNX (board={board_size}x{board_size}, opset={opset}) ...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={"board": {0: "batch"}, "policy_logits": {0: "batch"}, "value": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  -> ONNX written: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")


def shard_file(
    input_path: str,
    output_dir: str,
    num_shards: int,
    base_name: str = "gobot_model",
) -> list[dict]:
    """Split a file into N binary shards. Returns shard descriptors."""
    with open(input_path, "rb") as f:
        data = f.read()

    total_size = len(data)
    shard_size = math.ceil(total_size / num_shards)
    shards = []

    for i in range(num_shards):
        chunk = data[i * shard_size: (i + 1) * shard_size]
        if not chunk:
            break
        filename = f"{base_name}.shard{i:03d}.bin"
        shard_path = os.path.join(output_dir, filename)
        with open(shard_path, "wb") as f:
            f.write(chunk)

        shard_hash = hashlib.sha256(chunk).hexdigest()
        shards.append({
            "index": i,
            "filename": filename,
            "size": len(chunk),
            "sha256": shard_hash,
        })
        print(f"  Shard {i}: {filename} ({len(chunk) / 1024:.1f} KB)")

    return shards


def write_manifest(
    output_dir: str,
    shards: list[dict],
    onnx_path: str,
    model: GoResNet,
    num_shards: int,
) -> None:
    total_size = sum(s["size"] for s in shards)
    manifest = {
        "version": "1.0",
        "model_name": "GoBot Neural Go Engine",
        "board_size": model.board_size,
        "in_channels": model.in_channels,
        "num_filters": model.num_filters,
        "num_blocks": model.num_blocks,
        "action_size": model.action_size,
        "total_size_bytes": total_size,
        "total_size_kb": round(total_size / 1024, 1),
        "num_shards": len(shards),
        "shards": shards,
        "onnx_filename": os.path.basename(onnx_path) if num_shards == 1 else None,
        "inputs": [{"name": "board", "shape": [1, model.in_channels, model.board_size, model.board_size]}],
        "outputs": [
            {"name": "policy_logits", "shape": [1, model.action_size]},
            {"name": "value", "shape": [1, 1]},
        ],
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")


def verify_onnx(onnx_path: str, model: GoResNet) -> None:
    """Quick sanity check: run onnxruntime inference and compare to PyTorch."""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, model.in_channels, model.board_size, model.board_size).astype(np.float32)
        ort_out = sess.run(None, {"board": dummy})

        model.eval()
        with torch.no_grad():
            pt_out = model(torch.tensor(dummy))

        policy_diff = np.abs(ort_out[0] - pt_out[0].numpy()).max()
        value_diff = abs(ort_out[1][0, 0] - pt_out[1].item())

        print(f"  Verification: policy max diff={policy_diff:.2e}, value diff={value_diff:.2e}")
        assert policy_diff < 1e-3, f"Policy output mismatch: {policy_diff}"
        assert value_diff < 1e-3, f"Value output mismatch: {value_diff}"
        print("  [OK] ONNX output matches PyTorch output")
    except ImportError:
        print("  (onnxruntime not installed - skipping verification)")


def main():
    parser = argparse.ArgumentParser(description="Export GoBot model to ONNX for browser inference")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", type=str, default="docs/model", help="Output directory")
    parser.add_argument("--output-name", type=str, default="gobot_model", help="Base filename (no extension)")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards (1 = no sharding)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-verify", action="store_true", help="Skip onnxruntime verification")
    args = parser.parse_args()

    # --- Find & load model ---
    model_path = find_model(args.model)
    print(f"Loading model: {model_path}")
    model = GoResNet.load_checkpoint(model_path, device="cpu")
    model.eval()
    print(f"  Architecture: board={model.board_size}x{model.board_size}, "
          f"filters={model.num_filters}, blocks={model.num_blocks}")

    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, f"{args.output_name}.onnx")

    # --- Export ---
    export_to_onnx(model, onnx_path, opset=args.opset)

    # --- Verify ---
    if not args.no_verify:
        verify_onnx(onnx_path, model)

    # --- Shard ---
    if args.num_shards > 1:
        print(f"\nSharding into {args.num_shards} pieces ...")
        shards = shard_file(onnx_path, args.output_dir, args.num_shards, args.output_name)
    else:
        # Single "shard" = the whole file
        file_hash = sha256_file(onnx_path)
        shards = [{
            "index": 0,
            "filename": f"{args.output_name}.onnx",
            "size": os.path.getsize(onnx_path),
            "sha256": file_hash,
        }]

    # --- Manifest ---
    write_manifest(args.output_dir, shards, onnx_path, model, args.num_shards)

    total_kb = sum(s["size"] for s in shards) / 1024
    print(f"\n[OK] Done! {len(shards)} file(s), {total_kb:.1f} KB total -> {args.output_dir}/")


if __name__ == "__main__":
    main()
