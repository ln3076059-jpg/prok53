"""Create reproducible V10 repository, data, literature, and checkpoint audits."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from training.common import sha256_file
from training.dmd.v9_diagnostics import DEVELOPMENT_SUBJECTS, PROTECTED_HOLDOUTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v10_research"
MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".pkl"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace"
    ).strip()


def _package_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "torchvision", "ultralytics", "opencv-python", "scikit-learn", "fastapi"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def _cpu_name() -> str:
    if os.name != "nt":
        return platform.processor() or "NOT_VERIFIED"
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        return platform.processor() or "NOT_VERIFIED"


def environment_audit() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disks = {}
    for drive in ("C:\\", "D:\\"):
        if Path(drive).exists():
            usage = shutil.disk_usage(drive)
            disks[drive[:1]] = {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
    nvidia_smi = shutil.which("nvidia-smi")
    gpu = {
        "nvidia_smi_path": nvidia_smi,
        "cuda_gpu_available": False,
        "status": "NO_NVIDIA_RUNTIME_DETECTED" if not nvidia_smi else "NVIDIA_RUNTIME_PRESENT",
    }
    if nvidia_smi:
        probe = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        gpu["probe_returncode"] = probe.returncode
        gpu["probe_output"] = probe.stdout.strip()
        gpu["cuda_gpu_available"] = probe.returncode == 0 and bool(probe.stdout.strip())
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "disks": disks,
        "gpu": gpu,
        "packages": _package_versions(),
    }


def repository_audit() -> dict[str, Any]:
    status_text = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    status_lines = [line for line in status_text.splitlines() if line]
    return {
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "upstream": _git("rev-parse", "@{u}"),
        "head_matches_upstream": _git("rev-parse", "HEAD") == _git("rev-parse", "@{u}"),
        "working_tree_changes": status_lines,
        "user_owned_changes_preserved": [
            line for line in status_lines if line[3:].replace("\\", "/") == "prompt.md"
        ],
        "recent_commits": _git("log", "--oneline", "-12").splitlines(),
        "remote": _git("remote", "get-url", "origin"),
    }


def _checkpoint_paths() -> list[Path]:
    paths = []
    for root in (ROOT / "models", ROOT / "kaggle"):
        if root.is_dir():
            paths.extend(
                path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
            )
    root_yolo = ROOT / "yolo11n.pt"
    if root_yolo.is_file():
        paths.append(root_yolo)
    return sorted(set(paths))


def _metadata_candidates(path: Path) -> list[Path]:
    candidates = [
        path.with_suffix(".provenance.json"),
        path.parent / "args.yaml",
        path.parent / "report.json",
        path.parent / "plan.json",
        path.parent / "sha256.txt",
    ]
    current = path.parent
    while current != ROOT and ROOT in current.parents:
        candidates.extend(
            [
                current / "metadata.json",
                current / "model_lock.json",
                current / "dataset_lock.json",
                current / "environment.json",
            ]
        )
        current = current.parent
    return sorted({item for item in candidates if item.is_file()})


def model_inventory() -> dict[str, Any]:
    rows = []
    by_hash: defaultdict[str, list[str]] = defaultdict(list)
    for path in _checkpoint_paths():
        digest = sha256_file(path)
        relative = path.relative_to(ROOT).as_posix()
        by_hash[digest].append(relative)
        metadata = _metadata_candidates(path)
        if metadata:
            provenance = "LOCAL_METADATA_PRESENT_NOT_RIGHTS_APPROVAL"
        elif relative.startswith("models/auxiliary/") or relative in {
            "models/base/yolo11s.pt",
            "models/base/yolo11s-cls.pt",
            "yolo11n.pt",
        }:
            provenance = "PUBLIC_PRETRAINED_LINEAGE_ASSERTED_LOCALLY_LICENSE_NOT_REVERIFIED"
        else:
            provenance = "PROVENANCE_INCOMPLETE"
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "provenance_status": provenance,
                "metadata_files": [item.relative_to(ROOT).as_posix() for item in metadata],
                "architecture": "NOT_SAFELY_INSPECTED_FROM_PICKLE",
                "class_mapping": "SEE_LOCAL_METADATA_OR_MODEL_CONFIG",
                "load_executed": False,
            }
        )
    for row in rows:
        row["byte_identical_paths"] = by_hash[row["sha256"]]
    known = {
        "active_phone_detector": "models/active/v2/phone_detector.pt",
        "active_seatbelt_detector": "models/active/v2/seatbelt_detector.pt",
        "active_seatbelt_classifier": "models/active/v2/seatbelt_classifier.pt",
        "active_combined": "models/active/best.pt",
        "auxiliary_coco": "models/auxiliary/yolo11n.pt",
        "auxiliary_pose": "models/auxiliary/yolo11n-pose.pt",
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "known_runtime_roles": known,
        "runtime_config": "models/model_config_v2.yaml",
        "checkpoints": rows,
        "safety": {
            "pickle_load_performed": False,
            "checkpoint_download_performed": False,
            "production_activation_changed": False,
        },
    }


def dataset_manifest() -> dict[str, Any]:
    generalization = _read(ROOT / "reports" / "ai85_next" / "dataset_generalization.json")
    subjects = {}
    for subject_id in DEVELOPMENT_SUBJECTS:
        source = generalization["subjects"][subject_id]
        subjects[subject_id] = {
            "role": "REPEATEDLY_USED_DEVELOPMENT_ONLY",
            "class_distribution": source["class_distribution"],
            "views": source["views"],
        }
    protected_present = {
        subject_id: (ROOT / "dataset" / "external_dmd" / "subjects" / subject_id).is_dir()
        for subject_id in PROTECTED_HOLDOUTS
    }
    return {
        "schema_version": 1,
        "source": "VICOMTECH_DMD_LOCAL_INTAKE",
        "governance_record": "dataset/external_dmd/README.md",
        "rights_status": "PENDING_PROJECT_REVIEW",
        "rights_accepted": False,
        "canonical_eligible": False,
        "temporal_development_eligible": True,
        "development_subjects": sorted(DEVELOPMENT_SUBJECTS),
        "protected_holdouts": sorted(PROTECTED_HOLDOUTS),
        "protected_holdout_directories_present": protected_present,
        "protected_holdouts_accessed_in_v10": False,
        "new_approved_subjects": [],
        "independent_holdout": None,
        "generalization_status": "DATA_GENERALIZATION_BLOCKED",
        "subjects": subjects,
        "reviewed_phone_lane": {
            "path": "dataset/reviewed/v9_phone_diagnostic/139f2b6f223e",
            "role": "DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_TRAINING_READY",
            "human_reviewed_cases": 13,
            "valid_boxes": 12,
            "visible_phone_no_valid_box": 1,
        },
        "external_upload_allowed": False,
        "training_allowed_by_v10_audit": False,
    }


def literature_rows() -> list[dict[str, Any]]:
    not_verified = "NOT VERIFIED"
    return [
        {
            "work": "DualScope: Capturing Critical Spatial and Temporal Cues for Distracted Driving Activity Recognition",
            "authors": "Zhijie Qiu; Shuaibo Li; Laixin Zhang; Xuming Hu; Wei Ma",
            "venue_year": "AAAI 2026",
            "doi": "10.1609/aaai.v40i10.37812",
            "paper": "https://ojs.aaai.org/index.php/AAAI/article/view/37812",
            "code": "https://github.com/Qqqzzjsupr3me/DualScope",
            "checkpoint": "https://huggingface.co/Qqqzzj/DualScope",
            "license": "MIT on Hugging Face model card; code license NOT VERIFIED",
            "dataset_split": "Seven DDAR datasets; exact split details require paper reproduction",
            "main_metric": not_verified,
            "hardware_cost": not_verified,
            "reproducibility": "PARTIAL: eight checkpoint files are listed on the official model card",
            "roadwatch_use": "Behavior crops, frame differences, and multi-scale temporal cues",
            "domain_risk": "Paper classification metrics are not Roadwatch event metrics",
        },
        {
            "work": "Vision-Language Models Can Identify Distracted Driver Behavior from Naturalistic Videos",
            "authors": "Official repository: zahid-isu; paper author list NOT VERIFIED in this audit",
            "venue_year": "IEEE T-ITS 2024",
            "doi": "10.1109/TITS.2024.3381175",
            "paper": "https://doi.org/10.1109/TITS.2024.3381175",
            "code": "https://github.com/zahid-isu/DriveCLIP",
            "checkpoint": "Repository contains model_ckpt; exact artifact provenance NOT VERIFIED",
            "license": "MIT repository",
            "dataset_split": "DMD/StateFarm/SynDD1/SAM-DD; repository contains driver-profile splits",
            "main_metric": not_verified,
            "hardware_cost": "ViT-L/14 reported as best repository variant; exact cost NOT VERIFIED",
            "reproducibility": "PARTIAL: official repository states frame-based inference; separate VideoCLIP link",
            "roadwatch_use": "Frozen semantic feature comparator once checkpoint and disk budget are approved",
            "domain_risk": "Frame labels and VideoCLIP claims cannot be treated as event F1",
        },
        {
            "work": "Enhancing Driver Monitoring Systems Based on Novel Multi-Task Fusion Algorithm",
            "authors": "Romas Vijeikis et al.",
            "venue_year": "Sensors 2025",
            "doi": "10.3390/s25216799",
            "paper": "https://www.mdpi.com/1424-8220/25/21/6799",
            "code": not_verified,
            "checkpoint": not_verified,
            "license": "Open-access article; code/model license NOT VERIFIED",
            "dataset_split": "DMD 70/20/10 plus five-fold CV; robustness tests on AUC_V2 and SAMDD",
            "main_metric": "Attentiveness/classification metrics; exact comparable Roadwatch metric absent",
            "hardware_cost": not_verified,
            "reproducibility": "PAPER_ONLY",
            "roadwatch_use": "Reliability-aware BODY/FACE/HANDS and task fusion",
            "domain_risk": "gC-14 has BODY only; missing views require masks, not negative evidence",
        },
        {
            "work": "Towards Efficient Risky Driving Detection: A Benchmark and a Semi-Supervised Model",
            "authors": not_verified,
            "venue_year": "Sensors 2024",
            "doi": "10.3390/s24051386",
            "paper": "https://www.mdpi.com/1424-8220/24/5/1386",
            "code": not_verified,
            "checkpoint": not_verified,
            "license": "Open-access article; code/model/data license NOT VERIFIED",
            "dataset_split": "Bayonet-Drivers plus re-annotated State Farm; exact split NOT VERIFIED",
            "main_metric": "Object-detection mAP; not event F1",
            "hardware_cost": not_verified,
            "reproducibility": "PAPER_ONLY_LOCAL_REPRODUCTION_NOT_ESTABLISHED",
            "roadwatch_use": "Small-object feature pyramid and uncertainty-aware reviewed pseudo-labeling",
            "domain_risk": "Pseudo-labeling is blocked on repeatedly examined development cases",
        },
        {
            "work": "Seatbelt Detection Algorithm Improved with Lightweight Approach and Attention Mechanism",
            "authors": "Liankui Qiu; Jiankun Rao; Xiangzhe Zhao",
            "venue_year": "Applied Sciences 2024",
            "doi": "10.3390/app14083346",
            "paper": "https://www.mdpi.com/2076-3417/14/8/3346",
            "code": not_verified,
            "checkpoint": not_verified,
            "license": "Open-access article; code/checkpoint/data license NOT VERIFIED",
            "dataset_split": "Self-built seatbelt dataset; exact reusable split/data rights NOT VERIFIED",
            "main_metric": "Paper reports RST-Net accuracy 98.25%; not Roadwatch Macro-F1",
            "hardware_cost": "Paper reports pruning/parameter reductions; exact inference hardware NOT VERIFIED",
            "reproducibility": "PAPER_ONLY",
            "roadwatch_use": "Torso ROI and Triplet Attention after approved three-state labels",
            "domain_risk": "Private/self-built data and binary visibility assumptions differ from UNCERTAIN state",
        },
        {
            "work": "Real-time Detection of Seat Belt Usage in Overhead Traffic Surveillance Using YOLOv7",
            "authors": "Catur Edi Widodo; Kusworo Adi; Priyono Priyono; Aji Setiawan",
            "venue_year": "Applied Computer Science 2025",
            "doi": "10.35784/acs_7594",
            "paper": "https://ph.pollub.pl/index.php/acs/article/view/7594",
            "code": not_verified,
            "checkpoint": not_verified,
            "license": "CC BY 4.0 article",
            "dataset_split": "Day/night overhead images; unseen test set; reusable dataset NOT VERIFIED",
            "main_metric": "Paper reports mAP50 97.46%, F1 95.37%, seatbelt AP 93.40%",
            "hardware_cost": not_verified,
            "reproducibility": "PAPER_ONLY",
            "roadwatch_use": "Vehicle/windshield/occupant ROI hierarchy for future roadside data",
            "domain_risk": "Overhead roadside geometry differs from current in-cabin DMD evidence",
        },
    ]


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_reports(
    output: Path,
    repo: dict[str, Any],
    environment: dict[str, Any],
    dataset: dict[str, Any],
    models: dict[str, Any],
    papers: list[dict[str, Any]],
) -> None:
    dirty = ", ".join(f"`{line}`" for line in repo["working_tree_changes"]) or "none"
    disk_rows = [
        [name, f"{row['total_bytes'] / 2**30:.2f}", f"{row['free_bytes'] / 2**30:.2f}"]
        for name, row in environment["disks"].items()
    ]
    report01 = f"""# V10 repository and environment audit

- Branch: `{repo['branch']}`.
- HEAD/upstream: `{repo['head']}` / `{repo['upstream']}`; synchronized: `{repo['head_matches_upstream']}`.
- Working-tree changes before V10: {dirty}. The user-owned `prompt.md` change is preserved.
- CPU: `{environment['cpu']}`, {environment['physical_cores']} physical / {environment['logical_cores']} logical cores.
- RAM: {environment['ram_total_bytes'] / 2**30:.2f} GiB total; {environment['ram_available_bytes'] / 2**30:.2f} GiB available at audit.
- GPU: `{environment['gpu']['status']}`; CUDA GPU available: `{environment['gpu']['cuda_gpu_available']}`.
- Python: `{environment['python']}`; packages: `{environment['packages']}`.

{_markdown_table(['Drive', 'Total GiB', 'Free GiB'], disk_rows)}

The machine is suitable for artifact checks and lightweight CPU probes. Low RAM/disk and no NVIDIA
runtime block large checkpoint downloads, full video-model training, and honest local GPU benchmarks.
The last V9 verification reported 569 full tests passing; V10 checks are recorded separately after code changes.
"""
    paper_table = _markdown_table(
        ["Work", "Venue", "Code/checkpoint", "Dataset/split", "Metric scope", "Reproducibility", "Roadwatch feasibility/risk"],
        [
            [
                item["work"],
                f"{item['venue_year']}; DOI {item['doi']}",
                f"{item['code']} / {item['checkpoint']}; {item['license']}",
                item["dataset_split"],
                item["main_metric"],
                item["reproducibility"],
                f"{item['roadwatch_use']}; risk: {item['domain_risk']}",
            ]
            for item in papers
        ],
    )
    report02 = f"""# V10 paper review and reproducibility matrix

All reported paper values remain paper-level evidence. None is a Roadwatch result.

{paper_table}

Primary sources are linked directly in `literature_review.json`. DualScope is the only reviewed work with
an official model card listing task checkpoints. DriveCLIP has an MIT frame-based code repository and
driver split files, but its video implementation is separate. The remaining works lack locally verified
official code/checkpoints. Given current disk/GPU limits, V10 applies the lightweight transferable ideas:
fixed behavior regions, temporal residuals, explicit missing-view handling, and negative context evidence.
"""
    view_rows = []
    for subject_id, subject in dataset["subjects"].items():
        view_rows.append(
            [subject_id, subject["role"], ", ".join(view["view"] for view in subject["views"]), sum(subject["class_distribution"].values())]
        )
    report03 = f"""# V10 dataset and license audit

- DMD rights status: `{dataset['rights_status']}`.
- `RIGHTS_ACCEPTED={str(dataset['rights_accepted']).lower()}`; `CANONICAL_ELIGIBLE={str(dataset['canonical_eligible']).lower()}`.
- Local temporal development eligibility: `{dataset['temporal_development_eligible']}`.
- Protected subjects: `{', '.join(dataset['protected_holdouts'])}`; accessed in V10: `false`.
- New approved subjects: none; independent holdout: none.
- Generalization status: **{dataset['generalization_status']}**.
- External upload and V10 fine-tuning admission: blocked.

{_markdown_table(['Subject', 'Role', 'Views', 'Labeled frame samples'], view_rows)}

The 13-case reviewed phone lane remains diagnostic-only and contributes zero automatic training items.
No DMD video, personal data, or protected subject may be uploaded to Kaggle/Hugging Face under this audit.
"""
    provenance_counts: defaultdict[str, int] = defaultdict(int)
    for row in models["checkpoints"]:
        provenance_counts[row["provenance_status"]] += 1
    active_rows = []
    by_path = {row["path"]: row for row in models["checkpoints"]}
    for role, path in models["known_runtime_roles"].items():
        item = by_path.get(path)
        active_rows.append(
            [role, path, item["bytes"] if item else "MISSING", item["sha256"] if item else "MISSING", item["provenance_status"] if item else "MISSING"]
        )
    report04 = f"""# V10 model checkpoint inventory

Inventory contains **{models['checkpoint_count']}** checkpoint files totaling
{models['total_bytes'] / 2**30:.2f} GiB. Checkpoints were hashed as bytes and were not deserialized.

{_markdown_table(['Runtime role', 'Path', 'Bytes', 'SHA-256', 'Provenance'], active_rows)}

Provenance-state counts: `{dict(sorted(provenance_counts.items()))}`.

`LOCAL_METADATA_PRESENT_NOT_RIGHTS_APPROVAL` means an args/report/metadata/lock file exists locally; it
does not prove dataset rights. Files without sufficient sidecars remain `PROVENANCE_INCOMPLETE`. Architecture
and class mapping are taken only from explicit configs/metadata, never guessed from a filename such as
`best.pt`. No checkpoint was downloaded, loaded, promoted, or activated.
"""
    for name, text in {
        "01_REPOSITORY_AND_ENVIRONMENT_AUDIT.md": report01,
        "02_PAPER_REVIEW_AND_REPRODUCIBILITY.md": report02,
        "03_DATASET_AND_LICENSE_AUDIT.md": report03,
        "04_MODEL_CHECKPOINT_INVENTORY.md": report04,
    }.items():
        (output / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def build_registry(output: Path, repo: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    experiments = [
        {
            "experiment_id": "V10-P0-AUDIT",
            "commit_sha": repo["head"],
            "hypothesis": "Local assets and governance can be verified without model loading or holdout access.",
            "independent_variable": "None; audit only",
            "controlled_variables": ["No checkpoint load", "No protected holdout read"],
            "dataset_hash": sha256_file(output / "dataset_manifest.json"),
            "license_status": dataset["rights_status"],
            "config": "models/model_config_v2.yaml",
            "training_seed": None,
            "compute": "CPU_AUDIT_ONLY",
            "predictions": [],
            "metrics": {},
            "limitations": ["Rights unaccepted", "No independent holdout"],
            "evidence_level": "LEVEL_1_DIAGNOSTIC",
            "decision": "KEEP",
        }
    ]
    representation_path = output / "representation_probe.json"
    if representation_path.is_file():
        representation = _read(representation_path)
        experiments.append(
            {
                "experiment_id": "V10-D0-D5-REPRESENTATION",
                "commit_sha": representation["git_head_before_experiment"],
                "hypothesis": representation["hypothesis"],
                "independent_variable": "Frozen representation",
                "controlled_variables": representation["controlled_variables"],
                "model_checkpoint_hash": "FROZEN_RESNET_HASH_RECORDED_IN_SOURCE_CACHE_METADATA",
                "dataset_hash": representation["dataset_hash"],
                "split_definition": "OUTER_LOSO_THREE_REPEATED_DEVELOPMENT_SUBJECTS",
                "license_status": "RIGHTS_ACCEPTED_FALSE_TEMPORAL_DEVELOPMENT_ELIGIBLE_TRUE",
                "training_seed": representation["controlled_variables"]["seed"],
                "compute": representation["environment"],
                "predictions": [
                    fold["prediction_path"]
                    for item in representation["representations"]
                    for fold in item["folds"]
                ],
                "metrics": {
                    item["representation_id"]: item["pooled"]
                    for item in representation["representations"]
                },
                "runtime_seconds": representation["runtime_seconds"],
                "limitations": representation["limitations"],
                "evidence_level": representation["evidence_level"],
                "decision": "REJECT",
                "decision_detail": representation["model_selection_decision"],
            }
        )
    root_cause_path = output / "fn_root_cause.json"
    if root_cause_path.is_file():
        root_cause = _read(root_cause_path)
        experiments.append(
            {
                "experiment_id": "V10-PB-13FN-ROOT-CAUSE",
                "commit_sha": repo["head"],
                "hypothesis": "The frozen 13 phone FNs contain failures beyond detector misses.",
                "independent_variable": "None; evidence consolidation",
                "controlled_variables": ["Frozen ground truth", "Frozen matcher", "Frozen 13-case population"],
                "model_checkpoint_hash": None,
                "dataset_hash": root_cause["source_artifacts"],
                "split_definition": "THIRTEEN_SELECTED_DEVELOPMENT_FALSE_NEGATIVES",
                "license_status": dataset["rights_status"],
                "config": None,
                "training_seed": None,
                "compute": "CPU_ARTIFACT_AUDIT",
                "predictions": [],
                "metrics": root_cause["summary"],
                "runtime_seconds": None,
                "limitations": [
                    "Selected false negatives are diagnostic-only",
                    "No verified occupant tracks",
                    "Candidate yield is not event recall",
                ],
                "evidence_level": root_cause["evidence_level"],
                "decision": "KEEP",
            }
        )
    replay_path = ROOT / "reports" / "v9_research" / "phone_context_event_replay.json"
    if replay_path.is_file():
        replay = _read(replay_path)
        experiments.append(
            {
                "experiment_id": "V10-G3-INHERITED-CONTEXT-REPLAY",
                "commit_sha": repo["head"],
                "hypothesis": "Mounted/static negative context removes false alarms without losing TP.",
                "independent_variable": "Stored phone context admission",
                "controlled_variables": replay["comparison"],
                "model_checkpoint_hash": None,
                "dataset_hash": sha256_file(replay_path),
                "split_definition": "TWO_REPEATED_DEVELOPMENT_SUBJECTS",
                "license_status": dataset["rights_status"],
                "config": replay["temporal_config_sha256"],
                "training_seed": None,
                "compute": "CACHED_EVENT_REPLAY",
                "predictions": [],
                "metrics": replay["aggregate"],
                "runtime_seconds": None,
                "limitations": replay["limitations"],
                "evidence_level": "LEVEL_2_DEVELOPMENT_ABLATION",
                "decision": "INCONCLUSIVE",
            }
        )
    v9_path = ROOT / "reports" / "v9_research" / "v9_diagnostics.json"
    if v9_path.is_file():
        v9 = _read(v9_path)
        experiments.append(
            {
                "experiment_id": "V10-H-INHERITED-SEATBELT-FROZEN",
                "commit_sha": repo["head"],
                "hypothesis": "Frozen ROI/TTA variants may improve three-state seatbelt classification.",
                "independent_variable": "Resolution and horizontal-flip TTA",
                "controlled_variables": ["Frozen checkpoint", "Frozen labels", "Three-state semantics"],
                "model_checkpoint_hash": "SEE_MODEL_MANIFEST_ACTIVE_SEATBELT_CLASSIFIER",
                "dataset_hash": sha256_file(v9_path),
                "split_definition": "INHERITED_DEVELOPMENT_SPLIT",
                "license_status": dataset["rights_status"],
                "config": "INHERITED_V9_FROZEN_VARIANTS",
                "training_seed": None,
                "compute": "FROZEN_INFERENCE",
                "predictions": [],
                "metrics": v9["seatbelt"]["frozen_variants"],
                "runtime_seconds": None,
                "limitations": ["500 labels remain pending", "No independent holdout"],
                "evidence_level": "LEVEL_2_DEVELOPMENT_ABLATION",
                "decision": "BLOCKED",
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "production_approved": False,
        "experiments": experiments,
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    repo = repository_audit()
    environment = environment_audit()
    dataset = dataset_manifest()
    models = model_inventory()
    papers = literature_rows()
    _write_json(output / "repository_environment.json", {"repository": repo, "environment": environment})
    _write_json(output / "dataset_manifest.json", dataset)
    _write_json(output / "model_manifest.json", models)
    _write_json(output / "literature_review.json", {"schema_version": 1, "works": papers})
    render_reports(output, repo, environment, dataset, models, papers)
    registry = build_registry(output, repo, dataset)
    _write_json(output / "experiment_registry.json", registry)
    return {
        "reports": 4,
        "checkpoints": models["checkpoint_count"],
        "rights_accepted": dataset["rights_accepted"],
        "gpu_available": environment["gpu"]["cuda_gpu_available"],
        "head": repo["head"],
        "upstream": repo["upstream"],
        "user_changes_preserved": repo["user_owned_changes_preserved"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
