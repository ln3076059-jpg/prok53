from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

from training.common import sha256_file, stable_json_hash
from training.epoch_snapshots import (
    check_disk_space_for_snapshots,
    create_snapshots_manifest,
    save_epoch_snapshot,
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _build_recovery_archive(
    working: Path,
    experiment_id: str,
    config_path: Path,
    manifest: Path,
    reports: dict[str, dict],
) -> Path:
    """Atomically rebuild the portable recovery archive from completed components."""
    destination = working / f"{experiment_id}_RECOVERY.zip"
    with tempfile.TemporaryDirectory(prefix="v2-recovery-", dir=working) as temporary:
        temporary_root = Path(temporary)
        archive_root = temporary_root / f"{experiment_id}_RECOVERY"
        archive_root.mkdir()
        shutil.copy2(config_path, archive_root / "config.yaml")
        shutil.copy2(manifest, archive_root / manifest.name)
        for name, report in reports.items():
            target = archive_root / name
            target.mkdir()
            shutil.copy2(report["best_weights"], target / "best.pt")
            run = Path(report["run"])
            for source, filename in (
                (run / "weights" / "last.pt", "last.pt"),
                (run / "results.csv", "results.csv"),
                (run / "args.yaml", "args.yaml"),
                (run / "v2_component_report.json", "report.json"),
                (Path(report["plan"]), "plan.json"),
            ):
                if not source.is_file():
                    raise FileNotFoundError(f"missing recovery artifact: {source}")
                shutil.copy2(source, target / filename)
        staged = Path(
            shutil.make_archive(
                str(temporary_root / f"{experiment_id}_RECOVERY"),
                "zip",
                archive_root,
            )
        )
        staged.replace(destination)
    return destination


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _write_resume_backup(
    run: Path,
    backup_dir: Path,
    run_name: str,
    plan_path: Path,
    plan_fingerprint: str,
    completed_epochs: int,
) -> Path:
    """Mirror the latest optimizer checkpoint to a second, optionally synced disk."""
    target = backup_dir / run_name
    last = run / "weights" / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(f"resume backup source is missing: {last}")
    for source, relative in (
        (last, Path("weights/last.pt")),
        (run / "weights" / "best.pt", Path("weights/best.pt")),
        (run / "results.csv", Path("results.csv")),
        (run / "args.yaml", Path("args.yaml")),
        (plan_path, Path("plan.json")),
    ):
        if source.is_file():
            _copy_atomic(source, target / relative)
    metadata = {
        "schema_version": 1,
        "run_name": run_name,
        "completed_epochs": completed_epochs,
        "plan_fingerprint": plan_fingerprint,
        "last_weights_sha256": sha256_file(target / "weights" / "last.pt"),
    }
    _write_json_atomic(target / "resume_metadata.json", metadata)
    return target


def _restore_resume_backup(
    backup_dir: Path,
    run: Path,
    run_name: str,
    plan_fingerprint: str,
) -> bool:
    source = backup_dir / run_name
    metadata_path = source / "resume_metadata.json"
    last = source / "weights" / "last.pt"
    if not metadata_path.is_file() and not last.is_file():
        return False
    if not metadata_path.is_file() or not last.is_file():
        raise FileNotFoundError(f"STOP: incomplete external resume backup: {source}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("plan_fingerprint") != plan_fingerprint:
        raise ValueError("STOP: external resume backup plan mismatch")
    if metadata.get("last_weights_sha256") != sha256_file(last):
        raise ValueError("STOP: external resume backup last.pt mismatch")
    for backup_source, destination in (
        (last, run / "weights" / "last.pt"),
        (source / "weights" / "best.pt", run / "weights" / "best.pt"),
        (source / "results.csv", run / "results.csv"),
        (source / "args.yaml", run / "args.yaml"),
    ):
        if backup_source.is_file():
            _copy_atomic(backup_source, destination)
    return True


def _resume_backup_callback(
    backup_dir: Path,
    run_name: str,
    plan_path: Path,
    plan_fingerprint: str,
):
    def callback(trainer) -> None:
        _write_resume_backup(
            Path(trainer.save_dir),
            backup_dir,
            run_name,
            plan_path,
            plan_fingerprint,
            int(trainer.epoch) + 1,
        )

    return callback


def assert_finite_training_state(trainer) -> None:
    import torch

    for name, value in (
        ("loss", getattr(trainer, "loss", None)),
        ("running_loss", getattr(trainer, "tloss", None)),
    ):
        if value is None:
            continue
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError):
            continue
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            raise FloatingPointError(f"STOP_NONFINITE: {name} contains inf/nan at epoch {epoch}")


def environment_report() -> dict:
    import torch
    import ultralytics

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    return report


def _runtime_detection_yaml(source: Path, working: Path, component: str) -> Path:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["path"] = str(source.parent.resolve())
    target = working / f"{component}_data.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def _validate_component(component: str, config: dict) -> None:
    required = {"task", "model", "data", "epochs", "imgsz", "device"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"{component} is missing keys: {sorted(missing)}")
    if config["task"] not in {"detect", "classify"}:
        raise ValueError(f"unsupported V2 task: {config['task']}")
    data = Path(config["data"])
    if config["task"] == "detect" and not data.is_file():
        raise FileNotFoundError(f"detector data YAML not found: {data}")
    if config["task"] == "classify" and not data.is_dir():
        raise FileNotFoundError(f"classifier dataset not found: {data}")
    if config["task"] == "classify":
        report_path = data / "CLASSIFIER_DATASET_REPORT.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"classifier dataset report not found: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("ready_for_training"):
            raise ValueError(
                "classifier dataset is not ready: every split needs FASTENED, UNFASTENED, "
                "and UNCERTAIN_OR_OCCLUDED samples approved under its declared governance level"
            )


def train_component(
    component: str,
    component_config: dict,
    experiment_id: str,
    seed: int,
    working: Path,
    resume: bool = False,
    backup_dir: Path | None = None,
    backup_epoch_snapshots: bool = False,
) -> dict:
    from ultralytics import YOLO

    _validate_component(component, component_config)
    config = dict(component_config)
    task = config.pop("task")
    model_name = config.pop("model")
    data = Path(config.pop("data"))
    if task == "detect":
        data_argument = _runtime_detection_yaml(data, working, component)
    else:
        data_argument = data.resolve()
    run_name = f"{experiment_id}_{component}"
    config.update(
        data=str(data_argument),
        seed=seed,
        project=str(working),
        name=run_name,
        exist_ok=False,
    )
    run = working / run_name
    plan = {
        "schema_version": 1,
        "component": component,
        "experiment_id": experiment_id,
        "task": task,
        "model": str(model_name),
        "data": str(Path(data_argument).resolve()),
        "data_sha256": sha256_file(Path(data_argument)),
        "training": config,
    }
    plan["fingerprint"] = stable_json_hash(plan)
    plan_path = working / f"{run_name}_PLAN.json"
    if plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise ValueError(f"STOP: resume plan mismatch for {component}")
    else:
        _write_json_atomic(plan_path, plan)

    if backup_dir is not None and resume and not run.exists():
        _restore_resume_backup(backup_dir, run, run_name, plan["fingerprint"])

    report_path = run / "v2_component_report.json"
    best = run / "weights" / "best.pt"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("plan_fingerprint") != plan["fingerprint"]:
            raise ValueError(f"STOP: completed report plan mismatch for {component}")
        if not best.is_file() or sha256_file(best) != report.get("best_weights_sha256"):
            raise ValueError(f"STOP: completed best.pt mismatch for {component}")
        create_snapshots_manifest(run, component, plan["fingerprint"])
        return {**report, "reused_completed": True}

    resumed = False
    skipped_completed_checkpoint = False

    def _epoch_snapshot_callback(trainer):
        save_epoch_snapshot(
            trainer=trainer,
            component=component,
            plan_fingerprint=plan["fingerprint"],
            run_dir=run,
            backup_dir=backup_dir,
            backup_epoch_snapshots=backup_epoch_snapshots,
        )

    if run.exists():
        last = run / "weights" / "last.pt"
        if not resume:
            raise FileExistsError(
                f"run already exists: {run}; pass resume=True only after verifying the plan"
            )
        if not last.is_file():
            raise FileNotFoundError(f"STOP: incomplete run has no resumable last.pt: {last}")
        model = YOLO(str(last))
        model.add_callback("on_train_batch_end", assert_finite_training_state)
        model.add_callback("on_model_save", _epoch_snapshot_callback)
        if backup_dir is not None:
            model.add_callback(
                "on_model_save",
                _resume_backup_callback(backup_dir, run_name, plan_path, plan["fingerprint"]),
            )
        completed_epochs = int((model.ckpt or {}).get("epoch", -1)) + 1
        total_epochs = int(config["epochs"])
        
        current_last_pt_size = last.stat().st_size
        check_disk_space_for_snapshots(run, current_last_pt_size, total_epochs)

        if completed_epochs >= total_epochs:
            skipped_completed_checkpoint = True
        else:
            model.train(resume=True)
            resumed = True
    else:
        model = YOLO(model_name)
        model.add_callback("on_train_batch_end", assert_finite_training_state)
        model.add_callback("on_model_save", _epoch_snapshot_callback)
        if backup_dir is not None:
            model.add_callback(
                "on_model_save",
                _resume_backup_callback(backup_dir, run_name, plan_path, plan["fingerprint"]),
            )
        model.train(**config)

    best = run / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training did not produce {best}")
    validation = YOLO(str(best)).val(
        data=str(data_argument),
        split="val",
        imgsz=int(config["imgsz"]),
        device=config["device"],
        verbose=False,
    )
    metrics = {key: float(value) for key, value in validation.results_dict.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"STOP_NONFINITE: validation metrics for {component}")
    report = {
        "component": component,
        "task": task,
        "run": str(run.resolve()),
        "best_weights": str(best.resolve()),
        "best_weights_sha256": sha256_file(best),
        "plan": str(plan_path.resolve()),
        "plan_fingerprint": plan["fingerprint"],
        "resumed": resumed,
        "skipped_completed_checkpoint": skipped_completed_checkpoint,
        "validation_metrics": metrics,
        "test_split_used": False,
    }
    create_snapshots_manifest(run, component, plan["fingerprint"])
    _write_json_atomic(run / "v2_component_report.json", report)
    return report


def train_suite(
    config_path: Path,
    working: Path,
    only: set[str] | None = None,
    resume: bool = False,
    backup_dir: Path | None = None,
    backup_epoch_snapshots: bool = False,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment_id"])
    seed = int(config.get("seed", 42))
    components = config.get("components", {})
    if not components:
        raise ValueError("V2 suite has no components")
    selected = {name: value for name, value in components.items() if not only or name in only}
    if only and set(only) - set(components):
        raise ValueError(f"unknown components: {sorted(set(only) - set(components))}")
    working.mkdir(parents=True, exist_ok=True)
    state_path = working / f"{experiment_id}_STATE.json"
    reports: dict[str, dict] = {}
    for name, item in selected.items():
        _write_json_atomic(
            state_path,
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "status": "TRAINING",
                "active_component": name,
                "completed_components": sorted(reports),
                "remaining_components": [item for item in selected if item not in reports],
            },
        )
        reports[name] = train_component(
            name,
            item,
            experiment_id,
            seed,
            working,
            resume=resume,
            backup_dir=backup_dir,
            backup_epoch_snapshots=backup_epoch_snapshots,
        )
    suite = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "components": reports,
        "environment": environment_report(),
        "runtime_policy": config.get("runtime_policy"),
        "external_resume_backup": str(backup_dir.resolve()) if backup_dir else None,
        "policy": "VALIDATION_ONLY_DURING_TRAINING; FROZEN_TEST_AFTER_MODEL_LOCK",
    }
    manifest = working / f"{experiment_id}_MANIFEST.json"
    archive = working / f"{experiment_id}_RECOVERY.zip"
    if (
        manifest.is_file()
        and archive.is_file()
        and all(report.get("reused_completed") for report in reports.values())
    ):
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != sha256_file(config_path):
            raise ValueError("STOP: completed suite config mismatch")
        if set(existing.get("components", {})) != set(reports):
            raise ValueError("STOP: completed suite component mismatch")
        if existing.get("recovery_sha256") != sha256_file(archive):
            raise ValueError("STOP: completed recovery archive mismatch")
        return {**existing, "reused_completed_suite": True}

    _write_json_atomic(manifest, suite)
    archive = _build_recovery_archive(working, experiment_id, config_path, manifest, reports)
    suite["recovery_archive"] = str(archive.resolve())
    suite["recovery_sha256"] = sha256_file(archive)
    _write_json_atomic(manifest, suite)
    _write_json_atomic(
        state_path,
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "status": "COMPLETE",
            "active_component": None,
            "completed_components": sorted(reports),
            "remaining_components": [],
            "manifest": str(manifest.resolve()),
            "recovery_archive": str(archive.resolve()),
            "recovery_sha256": suite["recovery_sha256"],
        },
    )
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V2 specialist model suite")
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/MULTIMODEL_V2/config.yaml")
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Optional second disk/cloud-synced directory for per-epoch resume checkpoints",
    )
    parser.add_argument("--working", type=Path, default=Path("runs/v2"))
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional component names: phone_detector seatbelt_detector seatbelt_classifier",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the saved component plan matches and last.pt exists",
    )
    parser.add_argument(
        "--backup-epoch-snapshots",
        action="store_true",
        help="Mirror per-epoch snapshots to backup dir",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            train_suite(
                args.config,
                args.working,
                set(args.only or []),
                resume=args.resume,
                backup_dir=args.backup_dir,
                backup_epoch_snapshots=args.backup_epoch_snapshots,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
