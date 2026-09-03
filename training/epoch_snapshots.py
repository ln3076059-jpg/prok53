import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def check_disk_space_for_snapshots(
    run_dir: Path, current_last_pt_size: int, total_epochs: int
) -> None:
    """Checks if there's enough disk space for all projected epoch snapshots."""
    if current_last_pt_size == 0 or total_epochs == 0:
        return
        
    projected_bytes = current_last_pt_size * total_epochs
    projected_gb = projected_bytes / (1024**3)

    if projected_gb > 100:
        print(
            f"WARNING: Projected epoch snapshots size is {projected_gb:.2f} GB, which exceeds 100 GB."
        )

    # Check free disk space on the drive of the run directory
    check_path = run_dir.parent.absolute()
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent
    if not check_path.exists():
        check_path = Path.cwd()

    total, used, free = shutil.disk_usage(check_path)

    # if projected usage > 70% free disk: STOP and warn
    if projected_bytes > 0.7 * free:
        raise RuntimeError(
            f"STOP: Projected snapshot size ({projected_gb:.2f} GB) exceeds 70% of free disk space ({free / (1024**3):.2f} GB)."
        )


def save_epoch_snapshot(
    trainer,
    component: str,
    plan_fingerprint: str,
    run_dir: Path,
    backup_dir: Path | None = None,
    backup_epoch_snapshots: bool = False,
) -> None:
    """Callback function to save per-epoch snapshot from last.pt."""
    epoch = int(getattr(trainer, "epoch", -1)) + 1
    max_epochs = int(getattr(trainer, "epochs", 0))
    last_pt = run_dir / "weights" / "last.pt"

    if not last_pt.is_file():
        return

    snapshot_dir = run_dir / "epoch_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_pt = snapshot_dir / f"epoch_{epoch:03d}.pt"
    snapshot_json = snapshot_dir / f"epoch_{epoch:03d}.json"

    if snapshot_pt.exists() or snapshot_json.exists():
        if not snapshot_pt.exists() or not snapshot_json.exists():
            raise ValueError(f"STOP: Partial snapshot conflict for {snapshot_pt.name}")
        existing_metadata = json.loads(snapshot_json.read_text(encoding="utf-8"))
        if existing_metadata.get("plan_fingerprint") != plan_fingerprint:
            raise ValueError(f"STOP: Existing snapshot {snapshot_pt} has a different plan fingerprint.")
        if existing_metadata.get("checkpoint_sha256") != sha256_file(snapshot_pt):
            raise ValueError(f"STOP: Existing snapshot {snapshot_pt} SHA256 mismatch.")
        return

    _copy_atomic(last_pt, snapshot_pt)
    if snapshot_pt.stat().st_size == 0:
        raise RuntimeError(f"STOP: Created snapshot {snapshot_pt} is 0 bytes.")

    metrics = getattr(trainer, "metrics", {})
    loss_val = getattr(trainer, "loss", None)
    loss = None
    if loss_val is not None:
        try:
            # Ultralytics often returns a tensor for loss, or a dict
            import torch
            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.tolist()
            if isinstance(loss_val, (list, tuple)):
                loss = {f"loss_{i}": float(v) for i, v in enumerate(loss_val)}
            elif isinstance(loss_val, dict):
                loss = {str(k): float(v) for k, v in loss_val.items()}
            else:
                loss = {"loss": float(loss_val)}
        except Exception:
            pass

    learning_rate = None
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        try:
            learning_rate = {f"lr_{i}": param_group["lr"] for i, param_group in enumerate(optimizer.param_groups)}
        except Exception:
            pass

    gpu_info = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_allocated_bytes": torch.cuda.memory_allocated(0),
                "memory_reserved_bytes": torch.cuda.memory_reserved(0),
            }
    except ImportError:
        pass

    last_pt_sha256 = sha256_file(last_pt)

    metadata = {
        "schema_version": 1,
        "component": component,
        "epoch": epoch,
        "max_epochs": max_epochs,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "checkpoint": f"epoch_snapshots/{snapshot_pt.name}",
        "checkpoint_sha256": last_pt_sha256,
        "last_pt_sha256": last_pt_sha256,
        "plan_fingerprint": plan_fingerprint,
        "metrics": metrics if isinstance(metrics, dict) else {},
        "loss": loss,
        "learning_rate": learning_rate,
        "gpu": gpu_info,
        "resume_source": "last.pt",
        "test_split_used": False,
    }

    _write_json_atomic(snapshot_json, metadata)

    if backup_dir and backup_epoch_snapshots:
        backup_snapshot_dir = backup_dir / run_dir.name / "epoch_snapshots"
        backup_snapshot_dir.mkdir(parents=True, exist_ok=True)
        _copy_atomic(snapshot_pt, backup_snapshot_dir / snapshot_pt.name)
        _copy_atomic(snapshot_json, backup_snapshot_dir / snapshot_json.name)


def create_snapshots_manifest(run_dir: Path, component: str, plan_fingerprint: str) -> None:
    snapshot_dir = run_dir / "epoch_snapshots"
    if not snapshot_dir.is_dir():
        print("HISTORICAL_PER_EPOCH_SNAPSHOTS_NOT_AVAILABLE")
        return

    pt_files = sorted(snapshot_dir.glob("epoch_*.pt"))
    if not pt_files:
        print("HISTORICAL_PER_EPOCH_SNAPSHOTS_NOT_AVAILABLE")
        return

    snapshots = []
    epochs = []
    for pt in pt_files:
        try:
            epoch_num = int(pt.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
            
        json_file = pt.with_suffix(".json")
        if not json_file.exists():
            raise ValueError(f"Missing JSON metadata for {pt}")
            
        metadata = json.loads(json_file.read_text(encoding="utf-8"))
        if metadata.get("plan_fingerprint") != plan_fingerprint:
            raise ValueError(f"Manifest generation failed: Plan fingerprint mismatch for {pt}")

        pt_size = pt.stat().st_size
        if pt_size == 0:
            raise ValueError(f"Snapshot {pt} has size 0")
            
        pt_sha256 = sha256_file(pt)
        if metadata.get("checkpoint_sha256") != pt_sha256:
            raise ValueError(f"Manifest generation failed: SHA256 mismatch for {pt}")
        
        epochs.append(epoch_num)
        snapshots.append(
            {
                "epoch": epoch_num,
                "path": f"epoch_snapshots/{pt.name}",
                "sha256": pt_sha256,
                "bytes": pt_size,
            }
        )

    if not snapshots:
        return

    first_epoch = min(epochs)
    last_epoch = max(epochs)

    expected_epochs = set(range(first_epoch, last_epoch + 1))
    actual_epochs = set(epochs)
    missing = expected_epochs - actual_epochs
    if missing:
        raise ValueError(f"Missing epoch snapshots in manifest generation: {sorted(missing)}")

    manifest = {
        "schema_version": 1,
        "component": component,
        "count": len(snapshots),
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "plan_fingerprint": plan_fingerprint,
        "snapshots": snapshots,
    }

    manifest_path = run_dir / "epoch_snapshots_manifest.json"
    _write_json_atomic(manifest_path, manifest)


