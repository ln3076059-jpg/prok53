# V10 repository and environment audit

- Branch: `roadwatch-v4-candidate`.
- HEAD/upstream: `01c9161c2535206225e1165c0c0880874458b9eb` / `01c9161c2535206225e1165c0c0880874458b9eb`; synchronized: `True`.
- Working-tree changes before V10: ` M prompt.md`, `?? reports/v10_research/`, `?? tests/test_v10_research.py`, `?? training/dmd/v10_audit.py`, `?? training/dmd/v10_representation_probe.py`, `?? training/dmd/v10_root_cause.py`, `?? training/dmd/v10_synthesis.py`, `?? training/dmd/v10_test_report.py`, `?? training/dmd/verify_v10.py`. The user-owned `prompt.md` change is preserved.
- CPU: `AMD Ryzen 5 5500U with Radeon Graphics`, 6 physical / 12 logical cores.
- RAM: 5.86 GiB total; 0.77 GiB available at audit.
- GPU: `NO_NVIDIA_RUNTIME_DETECTED`; CUDA GPU available: `False`.
- Python: `3.11.9`; packages: `{'torch': '2.13.0', 'torchvision': '0.28.0', 'ultralytics': '8.3.187', 'opencv-python': '4.10.0.84', 'scikit-learn': '1.9.1', 'fastapi': '0.116.1'}`.

| Drive | Total GiB | Free GiB |
| --- | --- | --- |
| C | 174.15 | 1.01 |
| D | 301.18 | 5.94 |

The machine is suitable for artifact checks and lightweight CPU probes. Low RAM/disk and no NVIDIA
runtime block large checkpoint downloads, full video-model training, and honest local GPU benchmarks.
The last V9 verification reported 569 full tests passing; V10 checks are recorded separately after code changes.
