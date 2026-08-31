from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml
from PIL import Image

EXPERIMENT_ID = "MC_BOOTSTRAP_001"
USAGE = "PROPOSAL_MODEL_ONLY"
STATUS = "PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA"
CLASS_NAMES = ["phone", "seatbelt_fastened", "seatbelt_unfastened"]
RESUME_ARCHIVE = f"{EXPERIMENT_ID}_LATEST_RESUME.zip"
RESUME_ARCHIVE_ALIASES = {RESUME_ARCHIVE, f"{EXPERIMENT_ID}_RESUME.zip"}
RESUME_REQUIRED_MEMBERS = {
    "weights/last.pt",
    "weights/best.pt",
    "results.csv",
    "args.yaml",
    "resume_metadata.json",
}
RESUME_OPTIONAL_MEMBERS: set[str] = set()
TRAIN_KEYS = {
    "task", "imgsz", "epochs", "patience", "batch", "device", "workers", "cache", "amp",
    "optimizer", "cos_lr", "seed", "deterministic", "save", "save_period", "degrees",
    "translate", "scale", "hsv_h", "hsv_s", "hsv_v", "fliplr", "mosaic", "mixup",
    "classes", "close_mosaic", "multi_scale", "warmup_epochs", "weight_decay",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def locate_bundle(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    candidates = list(Path("/kaggle/input").rglob("MC_BOOTSTRAP_EXPECTED_HASHES.json"))
    if len(candidates) != 1:
        raise ValueError(f"STOP: expected one MC bootstrap bundle, found {len(candidates)}")
    return candidates[0].parent


def verify_hashes(root: Path) -> int:
    expected = json.loads((root / "MC_BOOTSTRAP_EXPECTED_HASHES.json").read_text(encoding="utf-8"))
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"STOP: bundle hash mismatch: {relative}")
    return len(expected)


def read_names(data: dict) -> list[str]:
    names = data.get("names", {})
    return list(names) if isinstance(names, list) else [names[index] for index in sorted(names)]


def validate_label(path: Path) -> Counter:
    counts = Counter()
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        if line in seen:
            raise ValueError(f"STOP: duplicate label line {path}:{line_number}")
        seen.add(line)
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"STOP: invalid label {path}:{line_number}")
        class_id = int(parts[0])
        if class_id not in {0, 1, 2}:
            raise ValueError(f"STOP: noncanonical class {path}:{line_number}")
        x, y, width, height = (float(value) for value in parts[1:])
        if width <= 0 or height <= 0 or x - width / 2 < 0 or x + width / 2 > 1:
            raise ValueError(f"STOP: invalid horizontal box {path}:{line_number}")
        if y - height / 2 < 0 or y + height / 2 > 1:
            raise ValueError(f"STOP: invalid vertical box {path}:{line_number}")
        counts[class_id] += 1
    return counts


def preflight(root: Path) -> dict:
    verified = verify_hashes(root)
    dataset = root / "dataset"
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    if read_names(data) != CLASS_NAMES or data.get("nc") != 3:
        raise ValueError("STOP: class order must be phone/fastened/unfastened")
    bootstrap = json.loads((dataset / "BOOTSTRAP_MANIFEST.json").read_text(encoding="utf-8"))
    if bootstrap.get("status") != STATUS:
        raise ValueError("STOP: dataset is not marked proposal-model-only")
    if any(bootstrap.get("component_overlap", {}).values()):
        raise ValueError("STOP: bootstrap components overlap splits")
    audit = json.loads((root / "audit_report.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("errors"):
        raise ValueError("STOP: bootstrap audit did not pass")
    if audit.get("cross_split_near_pairs"):
        raise ValueError("STOP: near duplicates cross splits")
    if any(audit.get("overlaps", {}).values()):
        raise ValueError("STOP: audit reports split overlap")

    manifest = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(manifest) != audit.get("manifest_samples"):
        raise ValueError("STOP: manifest sample count differs from audit")
    manifest_sha256 = sha256_file(dataset / "manifest.jsonl")
    audit_manifest_sha256 = stable_json_hash(manifest)
    if audit.get("manifest_sha256") != audit_manifest_sha256:
        raise ValueError("STOP: normalized manifest SHA differs from audit")
    if any(item.get("usage") != USAGE for item in manifest):
        raise ValueError("STOP: manifest usage policy mismatch")
    if any(item.get("human_review_status") != "PENDING" for item in manifest):
        raise ValueError("STOP: bootstrap contains purported human approvals")
    for key in ("sha256", "base_group_id", "effective_group_id", "near_cluster_id"):
        seen: defaultdict[str, set[str]] = defaultdict(set)
        for item in manifest:
            seen[str(item[key])].add(item["split"])
        if any(len(splits) > 1 for splits in seen.values()):
            raise ValueError(f"STOP: {key} crosses splits")

    splits = {}
    for split in ("train", "val", "test"):
        images = sorted(path for path in (dataset / "images" / split).glob("*") if path.is_file())
        labels = sorted(path for path in (dataset / "labels" / split).glob("*.txt") if path.is_file())
        if not images or len(images) != len(labels):
            raise ValueError(f"STOP: invalid image/label count in {split}")
        instances = Counter()
        for image_path in images:
            with Image.open(image_path) as image:
                image.verify()
            label_path = dataset / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"STOP: missing label: {label_path.relative_to(root)}")
            instances.update(validate_label(label_path))
        if any(instances[class_id] == 0 for class_id in range(3)):
            raise ValueError(f"STOP: {split} is missing a canonical class")
        expected = audit["splits"][split]
        named_instances = {CLASS_NAMES[key]: instances[key] for key in range(3)}
        if expected["images"] != len(images) or expected["instances"] != named_instances:
            raise ValueError(f"STOP: {split} differs from audited counts")
        splits[split] = {"images": len(images), "instances": named_instances}
    return {
        "verified_files": verified,
        "splits": splits,
        "manifest_samples": len(manifest),
        "manifest_sha256": manifest_sha256,
        "audit_manifest_sha256": audit_manifest_sha256,
        "config_sha256": sha256_file(root / "config.yaml"),
        "expected_hashes_sha256": sha256_file(root / "MC_BOOTSTRAP_EXPECTED_HASHES.json"),
    }


def _safe_resume_names(archive: zipfile.ZipFile) -> set[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("STOP: duplicate member in resume archive")
    allowed = RESUME_REQUIRED_MEMBERS | RESUME_OPTIONAL_MEMBERS
    normalized = set()
    for name in names:
        path = Path(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"STOP: unsafe resume archive path: {name}")
        normalized_name = path.as_posix()
        if normalized_name not in allowed:
            raise ValueError(f"STOP: unexpected resume archive member: {name}")
        normalized.add(normalized_name)
    missing = RESUME_REQUIRED_MEMBERS - normalized
    if missing:
        raise ValueError(f"STOP: resume archive is incomplete: {sorted(missing)}")
    return normalized


def _validate_resume_metadata(
    metadata: dict,
    last_sha256: str,
    best_sha256: str,
    preflight_report: dict,
    total_epochs: int,
    source: Path,
) -> dict:
    required = {
        "experiment_id": EXPERIMENT_ID,
        "usage": USAGE,
        "status": STATUS,
        "config_sha256": preflight_report["config_sha256"],
        "manifest_sha256": preflight_report["manifest_sha256"],
        "audit_manifest_sha256": preflight_report["audit_manifest_sha256"],
        "expected_hashes_sha256": preflight_report["expected_hashes_sha256"],
        "total_epochs": total_epochs,
        "last_weights_sha256": last_sha256,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"STOP: resume metadata mismatch: {key}")
    epoch_completed = metadata.get("epoch_completed")
    if not isinstance(epoch_completed, int) or not 1 <= epoch_completed <= total_epochs:
        raise ValueError("STOP: invalid completed epoch in resume metadata")
    if metadata.get("best_weights_sha256") != best_sha256:
        raise ValueError("STOP: resume best.pt SHA mismatch")
    return {**metadata, "resume_source": str(source.resolve())}


def inspect_resume_archive(path: Path, preflight_report: dict, total_epochs: int) -> dict:
    try:
        with zipfile.ZipFile(path) as archive:
            _safe_resume_names(archive)
            metadata = json.loads(archive.read("resume_metadata.json"))
            last_sha256 = sha256_stream(archive.open("weights/last.pt"))
            best_sha256 = sha256_stream(archive.open("weights/best.pt"))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"STOP: unreadable resume archive: {path}") from error
    return _validate_resume_metadata(
        metadata, last_sha256, best_sha256, preflight_report, total_epochs, path
    )


def inspect_resume_directory(path: Path, preflight_report: dict, total_epochs: int) -> dict:
    missing = [name for name in RESUME_REQUIRED_MEMBERS if not (path / name).is_file()]
    if missing:
        raise ValueError(f"STOP: extracted resume directory is incomplete: {sorted(missing)}")
    try:
        metadata = json.loads((path / "resume_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"STOP: unreadable extracted resume metadata: {path}") from error
    return _validate_resume_metadata(
        metadata,
        sha256_file(path / "weights" / "last.pt"),
        sha256_file(path / "weights" / "best.pt"),
        preflight_report,
        total_epochs,
        path,
    )


def inspect_resume_source(path: Path, preflight_report: dict, total_epochs: int) -> dict:
    if path.is_dir():
        return inspect_resume_directory(path, preflight_report, total_epochs)
    return inspect_resume_archive(path, preflight_report, total_epochs)


def locate_resume_archives(explicit: Path | None, working: Path) -> list[Path]:
    if explicit:
        source = explicit.parent if explicit.name == "resume_metadata.json" else explicit
        if not source.exists():
            raise FileNotFoundError(f"STOP: resume source not found: {explicit}")
        return [source.resolve()]
    candidates = [
        path
        for path in Path("/kaggle/input").rglob("*.zip")
        if path.name in RESUME_ARCHIVE_ALIASES
    ]
    candidates.extend(path.parent for path in Path("/kaggle/input").rglob("resume_metadata.json"))
    local = working / RESUME_ARCHIVE
    if local.is_file():
        candidates.append(local)
    return sorted({path.resolve() for path in candidates}, key=str)


def restore_resume_archive(
    archives: list[Path], root: Path, working: Path, preflight_report: dict
) -> tuple[Path | None, dict | None]:
    if not archives:
        return None, None
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    total_epochs = int(config["epochs"])
    inspected = [inspect_resume_source(path, preflight_report, total_epochs) for path in archives]
    newest_epoch = max(item["epoch_completed"] for item in inspected)
    newest = [item for item in inspected if item["epoch_completed"] == newest_epoch]
    if len({item["last_weights_sha256"] for item in newest}) != 1:
        raise ValueError("STOP: conflicting resume archives at the newest epoch")
    selected = newest[0]
    selected_path = Path(selected["resume_source"])
    run = working / EXPERIMENT_ID
    run.mkdir(parents=True, exist_ok=True)
    (run / "weights").mkdir(parents=True, exist_ok=True)
    if selected_path.is_dir():
        for name in RESUME_REQUIRED_MEMBERS - {"resume_metadata.json"}:
            target = run / Path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected_path / name, target)
    else:
        with zipfile.ZipFile(selected_path) as archive:
            names = _safe_resume_names(archive)
            for name in names - {"resume_metadata.json"}:
                target = run / Path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
    restored_last = run / "weights" / "last.pt"
    if sha256_file(restored_last) != selected["last_weights_sha256"]:
        raise ValueError("STOP: restored last.pt SHA mismatch")
    return restored_last, selected


def write_resume_archive(trainer, working: Path, preflight_report: dict) -> Path:
    last = Path(trainer.last)
    best = Path(trainer.best)
    if not last.is_file():
        raise FileNotFoundError("STOP: Ultralytics did not write last.pt")
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "usage": USAGE,
        "status": STATUS,
        "epoch_completed": int(trainer.epoch) + 1,
        "total_epochs": int(trainer.epochs),
        "last_weights_sha256": sha256_file(last),
        "best_weights_sha256": sha256_file(best) if best.is_file() else None,
        "config_sha256": preflight_report["config_sha256"],
        "manifest_sha256": preflight_report["manifest_sha256"],
        "audit_manifest_sha256": preflight_report["audit_manifest_sha256"],
        "expected_hashes_sha256": preflight_report["expected_hashes_sha256"],
    }
    members = {
        "weights/last.pt": last,
        "weights/best.pt": best,
        "results.csv": Path(trainer.csv),
        "args.yaml": Path(trainer.save_dir) / "args.yaml",
    }
    missing = [name for name, source in members.items() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"STOP: cannot create resume archive; missing {missing}")
    target = working / RESUME_ARCHIVE
    temporary = working / f".{RESUME_ARCHIVE}.tmp"
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, source in members.items():
                archive.write(source, name)
            archive.writestr("resume_metadata.json", json.dumps(metadata, indent=2, sort_keys=True))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def environment(working: Path) -> dict:
    import torch
    import ultralytics

    if not torch.cuda.is_available():
        raise RuntimeError("STOP: Kaggle GPU is unavailable")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "ultralytics": ultralytics.__version__,
        "disk_free_bytes": shutil.disk_usage(working).free,
        "pip_freeze": subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout.splitlines(),
    }


def runtime_data_yaml(root: Path, working: Path) -> Path:
    data = yaml.safe_load((root / "dataset" / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str((root / "dataset").resolve())
    target = working / "mc_bootstrap_data.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def train(
    root: Path,
    working: Path,
    preflight_report: dict,
    resume_checkpoint: Path | None = None,
    resume_metadata: dict | None = None,
) -> tuple[Path, dict]:
    from ultralytics import YOLO

    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("STOP: config experiment_id mismatch")
    unknown = set(config) - TRAIN_KEYS - {"experiment_id", "model"}
    if unknown:
        raise ValueError(f"STOP: unapproved training keys: {sorted(unknown)}")
    model_name = config.pop("model")
    config.pop("experiment_id")
    config = {key: value for key, value in config.items() if key in TRAIN_KEYS}
    data_yaml = runtime_data_yaml(root, working)
    config.update(data=str(data_yaml), project=str(working), name=EXPERIMENT_ID, exist_ok=False)
    run = working / EXPERIMENT_ID
    if resume_checkpoint and resume_metadata and resume_metadata["epoch_completed"] >= config["epochs"]:
        best = run / "weights" / "best.pt"
        metrics = YOLO(best).val(
            data=str(data_yaml),
            split="test",
            imgsz=int(config["imgsz"]),
            device=config["device"],
            verbose=False,
        )
        test_metrics = {key: float(value) for key, value in metrics.results_dict.items()}
        (run / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        return run, test_metrics
    model = YOLO(resume_checkpoint if resume_checkpoint else model_name)
    model.add_callback(
        "on_model_save", lambda trainer: write_resume_archive(trainer, working, preflight_report)
    )
    if resume_checkpoint:
        model.train(
            resume=str(resume_checkpoint),
            data=str(data_yaml),
            imgsz=config["imgsz"],
            batch=config["batch"],
            device=config["device"],
            workers=config["workers"],
            close_mosaic=config["close_mosaic"],
        )
    else:
        model.train(**config)
    best = run / "weights" / "best.pt"
    metrics = YOLO(best).val(
        data=str(data_yaml), split="test", imgsz=int(config["imgsz"]), device=config["device"], verbose=False
    )
    test_metrics = {key: float(value) for key, value in metrics.results_dict.items()}
    (run / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return run, test_metrics


def package(
    run: Path,
    preflight_report: dict,
    env: dict,
    test_metrics: dict,
    working: Path,
    resumed_from: dict | None,
) -> Path:
    for relative in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "test_metrics.json"):
        if not (run / relative).is_file():
            raise FileNotFoundError(f"missing training artifact: {relative}")
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "usage": USAGE,
        "best_weights_sha256": sha256_file(run / "weights" / "best.pt"),
        "preflight": preflight_report,
        "test_metrics": test_metrics,
        "environment": env,
        "resumed_from": resumed_from,
        "epoch_checkpoint_count": len(list((run / "weights").glob("epoch*.pt"))),
    }
    (run / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    recovery_members = [
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "args.yaml",
        "test_metrics.json",
        "run_metadata.json",
    ]
    target = working / f"{EXPERIMENT_ID}_RECOVERY.zip"
    temporary = working / f".{EXPERIMENT_ID}_RECOVERY.zip.tmp"
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for relative in recovery_members:
                archive.write(run / relative, relative)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed Kaggle three-class proposal bootstrap runner")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--working", type=Path, default=Path(os.environ.get("MC_BOOTSTRAP_WORKING", "/kaggle/working"))
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        help=f"Validated {RESUME_ARCHIVE} from a previous Kaggle session",
    )
    args = parser.parse_args()
    root = locate_bundle(args.bundle)
    report = preflight(root)
    print(json.dumps({"preflight": report}, indent=2))
    if args.preflight_only:
        return
    args.working.mkdir(parents=True, exist_ok=True)
    resume_checkpoint, resume_metadata = restore_resume_archive(
        locate_resume_archives(args.resume, args.working), root, args.working, report
    )
    if resume_metadata:
        print(
            json.dumps(
                {
                    "resume_source": resume_metadata["resume_source"],
                    "resume_from_completed_epoch": resume_metadata["epoch_completed"],
                },
                indent=2,
            )
        )
    env = environment(args.working)
    run, test_metrics = train(root, args.working, report, resume_checkpoint, resume_metadata)
    archive = package(run, report, env, test_metrics, args.working, resume_metadata)
    print(json.dumps({"recovery_archive": str(archive), "sha256": sha256_file(archive)}, indent=2))


if __name__ == "__main__":
    main()
