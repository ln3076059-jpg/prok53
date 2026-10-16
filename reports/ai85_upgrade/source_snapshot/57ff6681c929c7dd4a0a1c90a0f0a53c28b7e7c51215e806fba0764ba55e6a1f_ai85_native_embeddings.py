"""Encode original driver ROIs directly at 112px with the inherited frozen ResNet.

The V5 cache discarded detail at 64px before upsampling to 112px. This extraction
changes only spatial sampling. Frame identities, timestamps, labels, ROI geometry,
ImageNet weights and normalization remain inherited. No download is performed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from training.dmd.ai85_audit import artifact, write_json
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.run_v5_experiments import discover_external_dmd_root


def encode_subject(subject: str, output: Path, batch_size: int = 16) -> dict:
    assert_not_consumed_holdout_for_development(subject, "ai85_native_embedding_extraction")
    if subject not in DEVELOPMENT_SUBJECTS:
        raise ValueError("only registered development subjects may be extracted")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root = discover_external_dmd_root()
    source_video = next((root / "subjects" / subject).glob("*_rgb_body.mp4"))
    old_cache = root / "derivatives/v5_rgb_cache" / f"v5_rgb_{subject}_64.npz"
    with np.load(old_cache, allow_pickle=False) as old:
        indices, timestamps, labels = old["frame_indices"], old["timestamps"], old["labels"]
    weights = ResNet18_Weights.IMAGENET1K_V1
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not weight_path.is_file():
        raise FileNotFoundError("the inherited local ImageNet weights are required")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    model.fc = nn.Identity()
    model.eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    output.mkdir(exist_ok=True, parents=True)
    destination = output / f"v5_resnet18_{subject}.npz"
    if destination.exists():
        raise FileExistsError(f"refusing to replace embeddings: {destination}")
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {source_video}")
    encoded, batch = [], []
    started = time.perf_counter()
    current = -1

    def flush() -> None:
        if not batch:
            return
        frames = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div_(255)
        with torch.inference_mode():
            encoded.append(model((frames - mean) / std).numpy())
        batch.clear()

    try:
        for index, target in enumerate(indices):
            while current < target:
                if not capture.grab():
                    raise ValueError(f"source ended before frame {target}")
                current += 1
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f"cannot decode frame {target}")
            h, w = frame.shape[:2]
            roi = frame[int(.10 * h):int(.95 * h), :int(.65 * w)]
            resized = cv2.resize(roi, (112, 112), interpolation=cv2.INTER_AREA)
            batch.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            if len(batch) == batch_size:
                flush()
            if (index + 1) % 256 == 0:
                print(f"native112 {subject}: {index + 1}/{len(indices)}", flush=True)
        flush()
    finally:
        capture.release()
    embeddings = np.concatenate(encoded).astype(np.float16)
    if len(embeddings) != len(labels):
        raise ValueError("embedding extraction lost frame alignment")
    np.savez_compressed(destination, embeddings=embeddings,
                        timestamps=timestamps.astype(np.float64), labels=labels,
                        frame_indices=indices)
    result = {
        "subject_id": subject, "records": len(labels), "image_size": 112,
        "roi_xyxy": [0., .10, .65, .95], "normalization": "IMAGENET1K_V1",
        "source_video": artifact(source_video), "source_cache": artifact(old_cache),
        "weights": artifact(weight_path), "output": artifact(destination),
        "extraction_seconds": time.perf_counter() - started,
        "labels_and_timestamps_inherited_without_changes": True,
    }
    write_json(output / f"{subject}_extraction.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=DEVELOPMENT_SUBJECTS, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/ai85_upgrade/native112/embeddings"))
    args = parser.parse_args()
    result = encode_subject(args.subject, args.output)
    print(f"extraction completed: {result['subject_id']} {result['extraction_seconds']:.2f}s")


if __name__ == "__main__":
    main()
