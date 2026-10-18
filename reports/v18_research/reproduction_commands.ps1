# Roadwatch V18 Reproduction Commands
# Run tests and verify all bug fixes
py -m pytest tests/test_v18_bug_fixes.py -v
py -m pytest tests/test_v17_recovery.py -v
py -m pytest tests/test_core.py -v

# Verify gA-1 dataset integrity
py -c "from training.dmd.adapter import parse_dmd_openlabel; from pathlib import Path; s = parse_dmd_openlabel(Path('dataset/dmd-dataset-distraction-gA-1/dmd/gA/1/s2/gA_1_s2_2019-03-08T09;21;03+01;00_rgb_ann_distraction.json')); print('Participant:', s.source_participant_id, 'Phone intervals:', len(s.phone_intervals))"
