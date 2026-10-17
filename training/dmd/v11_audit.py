"""Create V11 repository, environment, rights, literature, and checkpoint audits."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.common import sha256_file
from training.dmd.v10_audit import environment_audit, model_inventory

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v11_research"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace"
    ).strip()


def repository_audit() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).splitlines()
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "upstream": _git("rev-parse", "@{u}"),
        "head_matches_upstream_at_audit": _git("rev-parse", "HEAD") == _git("rev-parse", "@{u}"),
        "remote": _git("remote", "get-url", "origin"),
        "working_tree_changes": status,
        "user_owned_changes_preserved": [line for line in status if line[3:].replace("\\", "/") == "PRODUCT.md"],
        "recent_commits": _git("log", "--oneline", "-12").splitlines(),
        "automatic_push_or_main_merge_performed": False,
    }


def dataset_manifest() -> dict[str, Any]:
    prior = _read(ROOT / "reports" / "v10_research" / "dataset_manifest.json")
    prior.update(
        {
            "schema_version": 2,
            "audit_version": "V11",
            "protected_holdouts_accessed_in_v11": False,
            "new_approved_subjects": [],
            "independent_holdout": None,
            "generalization_status": "INDEPENDENT_AI85_NOT_ESTABLISHED",
            "external_upload_allowed": False,
            "training_allowed_by_v11_audit": False,
            "development_experiment_policy": "LIGHTWEIGHT_FEATURE_DIAGNOSTICS_ONLY",
            "selected_development_candidate": "E3_SPATIAL_PLUS_RUNNING",
            "selected_candidate_is_independent_evidence": False,
            "protected_holdout_directories_enumerated_or_opened_in_v11": False,
        }
    )
    prior.pop("training_allowed_by_v10_audit", None)
    return prior


def literature_rows() -> list[dict[str, Any]]:
    nv = "NOT VERIFIED"
    common = {
        "roadwatch_result": False,
        "local_reproduction_performed": False,
    }
    return [
        {
            **common,
            "title": "DualScope: Capturing Critical Spatial and Temporal Cues for Distracted Driving Activity Recognition",
            "authors": "Zhijie Qiu; Shuaibo Li; Laixin Zhang; Xuming Hu; Wei Ma",
            "year_venue": "AAAI 2026",
            "doi": "10.1609/aaai.v40i10.37812",
            "paper_url": "https://ojs.aaai.org/index.php/AAAI/article/view/37812",
            "dataset": "Eight checkpoint variants are listed; exact paper dataset matrix requires full reproduction",
            "evaluation_protocol": "Cross-dataset distracted-driving classification; exact split details not reproduced locally",
            "metrics_and_definition": "Classification metrics reported by paper; not Roadwatch event F1",
            "code": "https://github.com/Qqqzzjsupr3me/DualScope",
            "checkpoint": "https://huggingface.co/Qqqzzj/DualScope",
            "license": "MIT on model card; repository license not independently reverified",
            "transferable_method": "Critical-region pooling, dynamic differences, motion encoder, fine-to-coarse temporal pyramid",
            "estimated_cost": "Full reproduction needs GPU; lightweight causal feature analogue fits CPU",
            "roadwatch_difference": "Paper classification protocol differs from Roadwatch frozen event matching",
            "limitations": "No weights downloaded and no claimed reproduction under current rights/compute gates",
        },
        {
            **common,
            "title": "Vision-Language Models Can Identify Distracted Driver Behavior from Naturalistic Videos",
            "authors": "Md Zahid Hasan; Jaijing Chen; Jiyang Wang; Mohammed Shaiqur Rahman; Ameya Joshi",
            "year_venue": "IEEE Transactions on Intelligent Transportation Systems 2024",
            "doi": "10.1109/TITS.2024.3381175",
            "paper_url": "https://doi.org/10.1109/TITS.2024.3381175",
            "dataset": "DMD, StateFarm, SynDD1, SAM-DD according to official repository",
            "evaluation_protocol": "Repository provides driver-profile split files; local protocol not reproduced",
            "metrics_and_definition": "Frame/clip classification metrics; not Roadwatch event metrics",
            "code": "https://github.com/zahid-isu/DriveCLIP",
            "checkpoint": "Repository model_ckpt lineage not locally verified",
            "license": "MIT repository",
            "transferable_method": "Frozen CLIP embeddings, zero-shot prompts, task adapter",
            "estimated_cost": "ViT-L/14 memory and download exceed this low-RAM CPU audit budget",
            "roadwatch_difference": "Frame/clip labels and semantic prompts do not prove event recovery",
            "limitations": "D4 blocked by checkpoint, memory, and rights gates",
        },
        {
            **common,
            "title": "Enhancing Driver Monitoring Systems Based on Novel Multi-Task Fusion Algorithm",
            "authors": "Romas Vijeikis; Ibidapo Dare Dada; Adebayo A. Abayomi-Alli; Vidas Raudonis",
            "year_venue": "Sensors 2025",
            "doi": "10.3390/s25216799",
            "paper_url": "https://www.mdpi.com/1424-8220/25/21/6799",
            "dataset": "DMD plus AUC_V2 and SAMDD robustness evaluation",
            "evaluation_protocol": "DMD 70/20/10 and five-fold evaluation described by article",
            "metrics_and_definition": "Task classification/attentiveness metrics; not frozen Roadwatch Event F1",
            "code": nv,
            "checkpoint": nv,
            "license": "Open-access article; model/code license not verified",
            "transferable_method": "BODY/FACE/HANDS reliability masks and BiLSTM task fusion",
            "estimated_cost": "Moderate GPU training; lightweight reliability logic fits CPU",
            "roadwatch_difference": "Roadwatch has missing views and only three repeated development subjects",
            "limitations": "Missing camera evidence must remain UNKNOWN",
        },
        {
            **common,
            "title": "ActionFormer: Localizing Moments of Actions with Transformers",
            "authors": "Chenlin Zhang; Jianxin Wu; Yin Li",
            "year_venue": "ECCV 2022",
            "doi": nv,
            "paper_url": "https://arxiv.org/abs/2202.07925",
            "dataset": "Temporal action-localization benchmarks including THUMOS14 and ActivityNet",
            "evaluation_protocol": "Proposal-free multi-scale temporal localization",
            "metrics_and_definition": "Temporal mAP over IoU thresholds; distinct from Roadwatch matcher",
            "code": "https://github.com/happyharrycn/actionformer_release",
            "checkpoint": "Official repository lists pretrained models/configurations",
            "license": "MIT repository",
            "transferable_method": "Multi-scale causal candidates and separate boundary evidence",
            "estimated_cost": "Repository reports about 4.5 GB training VRAM and over 10 GB inference VRAM for THUMOS setup",
            "roadwatch_difference": "Roadwatch lacks an approved large event-training set",
            "limitations": "Full model blocked; only design ideas are adopted",
        },
        {
            **common,
            "title": "Domain Generalization with MixStyle",
            "authors": "Kaiyang Zhou; Yongxin Yang; Yu Qiao; Tao Xiang",
            "year_venue": "ICLR 2021",
            "doi": nv,
            "paper_url": "https://arxiv.org/abs/2104.02008",
            "dataset": "Multiple visual domain-generalization benchmarks",
            "evaluation_protocol": "Unseen-domain evaluation",
            "metrics_and_definition": "Classification accuracy on held-out domains",
            "code": "https://github.com/KaiyangZhou/mixstyle-release",
            "checkpoint": nv,
            "license": "MIT repository",
            "transferable_method": "Mix feature statistics to reduce cabin/style dependence",
            "estimated_cost": "Low incremental training cost, but still requires approved train data",
            "roadwatch_difference": "Only three repeatedly studied subjects are currently available",
            "limitations": "V11 uses deterministic normalization analogue, not a MixStyle reproduction",
        },
        {
            **common,
            "title": "Distributionally Robust Neural Networks for Group Shifts",
            "authors": "Shiori Sagawa; Pang Wei Koh; Tatsunori Hashimoto; Percy Liang",
            "year_venue": "ICLR 2020",
            "doi": nv,
            "paper_url": "https://arxiv.org/abs/1911.08731",
            "dataset": "Group-shift classification benchmarks",
            "evaluation_protocol": "Worst-group validation with regularization and early stopping",
            "metrics_and_definition": "Worst-group accuracy",
            "code": "https://github.com/kohpangwei/group_DRO",
            "checkpoint": nv,
            "license": "MIT repository",
            "transferable_method": "Worst-subject selection and group-aware regularization",
            "estimated_cost": "Low-to-moderate, but statistically weak with three subjects",
            "roadwatch_difference": "Roadwatch groups are repeatedly reused and too few for a strong Group DRO claim",
            "limitations": "Group DRO not run; worst-subject gating is used instead",
        },
        {
            **common,
            "title": "Seatbelt Detection Algorithm Improved with Lightweight Approach and Attention Mechanism",
            "authors": "Liankui Qiu; Jiankun Rao; Xiangzhe Zhao",
            "year_venue": "Applied Sciences 2024",
            "doi": "10.3390/app14083346",
            "paper_url": "https://www.mdpi.com/2076-3417/14/8/3346",
            "dataset": "Self-built seatbelt dataset",
            "evaluation_protocol": "Paper-specific train/test protocol not reproduced",
            "metrics_and_definition": "Paper reports 98.25% accuracy; not Roadwatch three-state Macro-F1",
            "code": nv,
            "checkpoint": nv,
            "license": "Open-access article; reusable data/model license not verified",
            "transferable_method": "Torso ROI, Triplet Attention, channel pruning",
            "estimated_cost": "Moderate training; blocked until labels and rights are approved",
            "roadwatch_difference": "Roadwatch explicitly preserves UNCERTAIN_OR_OCCLUDED",
            "limitations": "Binary visibility assumptions cannot relabel invisible belts as UNFASTENED",
        },
        {
            **common,
            "title": "Real-time Detection of Seat Belt Usage in Overhead Traffic Surveillance Using YOLOv7",
            "authors": "Catur Edi Widodo; Kusworo Adi; Priyono Priyono; Aji Setiawan",
            "year_venue": "Applied Computer Science 2025",
            "doi": "10.35784/acs_7594",
            "paper_url": "https://ph.pollub.pl/index.php/acs/article/view/7594",
            "dataset": "Day/night overhead road images with car, windshield, passenger, and seatbelt classes",
            "evaluation_protocol": "Paper-specific unseen test split",
            "metrics_and_definition": "Paper reports mAP50 97.46%, F1 95.37%, seatbelt AP 93.40%",
            "code": nv,
            "checkpoint": nv,
            "license": "CC BY 4.0 article; dataset/model license not verified",
            "transferable_method": "Hierarchical vehicle-to-windshield-to-occupant localization",
            "estimated_cost": "YOLOv7 GPU training; not justified locally",
            "roadwatch_difference": "Overhead roadside imagery differs from in-cabin DMD",
            "limitations": "Paper values cannot be transferred across domain or metric",
        },
    ]


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    repo = repository_audit()
    environment = environment_audit()
    dataset = dataset_manifest()
    models = model_inventory()
    previous = _read(ROOT / "reports" / "v10_research" / "model_manifest.json")
    old_hashes = {row["path"]: row["sha256"] for row in previous["checkpoints"]}
    mismatches = [
        row["path"]
        for row in models["checkpoints"]
        if row["path"] in old_hashes and row["sha256"] != old_hashes[row["path"]]
    ]
    models["v10_hash_reconciliation"] = {
        "previous_manifest": "reports/v10_research/model_manifest.json",
        "previous_manifest_sha256": sha256_file(ROOT / "reports" / "v10_research" / "model_manifest.json"),
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }
    papers = literature_rows()
    _write(output / "repository_environment.json", {"repository": repo, "environment": environment})
    _write(output / "dataset_manifest.json", dataset)
    _write(output / "model_manifest.json", models)
    _write(output / "literature_review.json", {"schema_version": 1, "papers": papers})
    return {
        "repository": repo,
        "environment": environment,
        "dataset": dataset,
        "models": models,
        "papers": papers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "head": result["repository"]["head"],
                "papers": len(result["papers"]),
                "checkpoints": result["models"]["checkpoint_count"],
                "rights_accepted": result["dataset"]["rights_accepted"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
