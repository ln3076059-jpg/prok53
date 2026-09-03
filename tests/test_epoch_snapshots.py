import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from training.epoch_snapshots import (
    check_disk_space_for_snapshots,
    create_snapshots_manifest,
    save_epoch_snapshot,
)
from training.common import sha256_file


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    
    last_pt = weights / "last.pt"
    last_pt.write_text("dummy model weights")
    return run


def test_naming_and_metadata_json(run_dir):
    trainer = Mock()
    trainer.epoch = 0 # 0-indexed, so it's epoch 1
    trainer.epochs = 100
    trainer.metrics = {"mAP50": 0.5}
    trainer.loss = [0.1, 0.2]
    
    save_epoch_snapshot(
        trainer,
        component="phone_detector",
        plan_fingerprint="abc",
        run_dir=run_dir,
    )
    
    snapshot_dir = run_dir / "epoch_snapshots"
    assert snapshot_dir.exists()
    assert (snapshot_dir / "epoch_001.pt").exists()
    assert (snapshot_dir / "epoch_001.json").exists()
    
    metadata = json.loads((snapshot_dir / "epoch_001.json").read_text())
    assert metadata["epoch"] == 1
    assert metadata["checkpoint"] == "epoch_snapshots/epoch_001.pt"
    assert metadata["metrics"] == {"mAP50": 0.5}
    assert metadata["loss"] == {"loss_0": 0.1, "loss_1": 0.2}


def test_atomic_copy_and_backup(run_dir, tmp_path):
    trainer = Mock()
    trainer.epoch = 9 # epoch 10
    trainer.epochs = 100
    
    backup_dir = tmp_path / "backup"
    
    save_epoch_snapshot(
        trainer,
        component="phone_detector",
        plan_fingerprint="abc",
        run_dir=run_dir,
        backup_dir=backup_dir,
        backup_epoch_snapshots=True,
    )
    
    assert (run_dir / "epoch_snapshots" / "epoch_010.pt").exists()
    assert (backup_dir / "run" / "epoch_snapshots" / "epoch_010.pt").exists()

    # With backup_epoch_snapshots=False
    trainer.epoch = 99 # epoch 100
    save_epoch_snapshot(
        trainer,
        component="phone_detector",
        plan_fingerprint="abc",
        run_dir=run_dir,
        backup_dir=backup_dir,
        backup_epoch_snapshots=False,
    )
    assert (run_dir / "epoch_snapshots" / "epoch_100.pt").exists()
    assert not (backup_dir / "run" / "epoch_snapshots" / "epoch_100.pt").exists()


def test_resume_no_overwrite_and_conflict(run_dir):
    # Setup existing snapshot 1
    trainer = Mock()
    trainer.epoch = 0
    trainer.epochs = 100
    save_epoch_snapshot(trainer, "comp", "abc", run_dir)
    
    pt_file = run_dir / "epoch_snapshots" / "epoch_001.pt"
    json_file = run_dir / "epoch_snapshots" / "epoch_001.json"
    
    original_sha = sha256_file(pt_file)
    
    # Try to save again with same fingerprint - should not overwrite (we simulate by changing last.pt)
    (run_dir / "weights" / "last.pt").write_text("new weights")
    save_epoch_snapshot(trainer, "comp", "abc", run_dir)
    
    assert sha256_file(pt_file) == original_sha
    
    # Conflict: partial snapshot (pt exists, json missing)
    json_file.unlink()
    with pytest.raises(ValueError, match="Partial snapshot conflict"):
        save_epoch_snapshot(trainer, "comp", "abc", run_dir)
        
    # Recreate JSON to continue
    json_file.write_text(json.dumps({"plan_fingerprint": "abc", "checkpoint_sha256": original_sha}))
    
    # Conflict: plan fingerprint mismatch
    trainer.epoch = 0
    with pytest.raises(ValueError, match="different plan fingerprint"):
        save_epoch_snapshot(trainer, "comp", "diff", run_dir)
        
    # Conflict: SHA mismatch (if someone corrupted it but JSON has old SHA)
    # Edit the JSON manually
    metadata = json.loads(json_file.read_text())
    metadata["checkpoint_sha256"] = "wrongsha"
    json_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        save_epoch_snapshot(trainer, "comp", "abc", run_dir)


def test_manifest_generation_and_early_stopping(run_dir):
    snapshot_dir = run_dir / "epoch_snapshots"
    snapshot_dir.mkdir(parents=True)
    
    # Create epochs 1, 2, 3 (early stop at 3)
    for i in range(1, 4):
        pt = snapshot_dir / f"epoch_{i:03d}.pt"
        pt.write_text(f"weight {i}")
        
        json_file = snapshot_dir / f"epoch_{i:03d}.json"
        json_file.write_text(json.dumps({
            "plan_fingerprint": "abc",
            "checkpoint_sha256": sha256_file(pt)
        }))
        
    create_snapshots_manifest(run_dir, "comp", "abc")
    
    manifest_file = run_dir / "epoch_snapshots_manifest.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text())
    
    assert manifest["count"] == 3
    assert manifest["first_epoch"] == 1
    assert manifest["last_epoch"] == 3
    assert len(manifest["snapshots"]) == 3


def test_manifest_missing_epoch(run_dir):
    snapshot_dir = run_dir / "epoch_snapshots"
    snapshot_dir.mkdir(parents=True)
    
    # Create epochs 1, 3 (missing 2)
    pt1 = snapshot_dir / "epoch_001.pt"
    pt1.write_text("w1")
    (snapshot_dir / "epoch_001.json").write_text(json.dumps({"plan_fingerprint": "abc", "checkpoint_sha256": sha256_file(pt1)}))
    
    pt3 = snapshot_dir / "epoch_003.pt"
    pt3.write_text("w3")
    (snapshot_dir / "epoch_003.json").write_text(json.dumps({"plan_fingerprint": "abc", "checkpoint_sha256": sha256_file(pt3)}))
    
    with pytest.raises(ValueError, match="Missing epoch snapshots"):
        create_snapshots_manifest(run_dir, "comp", "abc")


def test_disk_projected_usage_guard(run_dir):
    with patch('shutil.disk_usage') as mock_usage:
        # Mock free space to be 100 GB
        free_space = 100 * (1024**3)
        mock_usage.return_value = (0, 0, free_space)
        
        # 1 GB per epoch, 80 epochs = 80 GB projected
        # 70% of 100 GB is 70 GB. 80 GB > 70 GB -> STOP
        with pytest.raises(RuntimeError, match="exceeds 70% of free disk space"):
            check_disk_space_for_snapshots(run_dir, 1 * (1024**3), 80)
            
        # 0.5 GB per epoch, 100 epochs = 50 GB. 50 GB < 70 GB -> OK
        check_disk_space_for_snapshots(run_dir, 500 * (1024**2), 100)

