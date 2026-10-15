# Data and Artifact Inventory

## Development data

- Subjects used: `gB-9, gZ-36, gC-14`.
- Consumed holdouts forbidden from tuning: `gZ-37, gE-28`.
- Holdouts accessed by V6 runner: `[]`.
- Event truth: 29 phone-use events across the three development subjects.
- RGB caches and frozen ResNet embeddings are local under `dataset/external_dmd/derivatives`.
- No additional licensed development subjects were available; repeated iteration on three subjects is explicitly a limitation.

## Artifacts verified on disk

| Path | Exists | Bytes | SHA-256 |
|---|---:|---:|---|
| `models/active/best.pt` | yes | 19,209,626 | `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54` |
| `models/active/v2/phone_detector.pt` | yes | 19,209,626 | `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54` |
| `models/active/v2/seatbelt_detector.pt` | yes | 19,345,306 | `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500` |
| `models/active/v2/seatbelt_classifier.pt` | yes | 11,028,162 | `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4` |
| `models/active/v2/windshield_detector.pt` | no | 0 | — |
| `models/fusion/no_seatbelt_v2.json` | no | 0 | — |
| `models/candidate/v5_resnet18_gru_001/loso_gB-9.pt` | yes | 1,102,111 | `4578e9f1b148a8724966eaa1e2b6f1b2e5d798b16443f07f383b29affd1e82a1` |
| `models/candidate/v5_resnet18_gru_001/loso_gC-14.pt` | yes | 1,102,133 | `72e5ab0e7b8d47b99fca558b7aa0b3462a6b4939c0e7811b2d3e283475d8cef9` |
| `models/candidate/v5_resnet18_gru_001/loso_gZ-36.pt` | yes | 1,102,133 | `cf3b9fc414766976cf08d3c59f5809ebcb9c48fe8680a26f62465afe7c22876f` |
| `models/candidate/v5_tiny_temporal_cnn_001/loso_gB-9.pt` | yes | 76,118 | `5374cdc989b920ef0724807d82f3177af2b7ce7f8224e206a2af55edd2abc881` |
| `models/candidate/v5_tiny_temporal_cnn_001/loso_gC-14.pt` | yes | 76,149 | `107323111fc919c603b85c6c86091a040ac3a006f3b77dde1d79df8373406b55` |
| `models/candidate/v5_tiny_temporal_cnn_001/loso_gZ-36.pt` | yes | 76,149 | `74ee3d31a1c157c217ded6fe75a0a53e5add182be6f6154dfc5d0adc04ad7159` |
| `models/candidate/v6_action_ablation_001/flat_loss_only_loso_gB-9.pt` | yes | 1,103,145 | `e949f1e8c7d15b23cbb63db6d0fd8c17b35859e2938466f029c48cd0fd164e17` |
| `models/candidate/v6_action_ablation_001/flat_loss_only_loso_gC-14.pt` | yes | 1,103,167 | `289f578663c540f52f6f75dc3b059fd869c6885f3c4105fda03ce269a9b31a41` |
| `models/candidate/v6_action_ablation_001/flat_loss_only_loso_gZ-36.pt` | yes | 1,103,167 | `5ddecf7792dfc53a493184ae034730971b5082506b70606c91fb6deadf61e9fa` |
| `models/candidate/v6_action_ablation_001/flat_sampler_only_loso_gB-9.pt` | yes | 1,103,211 | `f608fc54bd4a1a0f2590bae6c839812abd49125e90c2e6752d877bb273f1530c` |
| `models/candidate/v6_action_ablation_001/flat_sampler_only_loso_gC-14.pt` | yes | 1,103,233 | `5e7f2fc1e4b74c1524aa4139228f9edfbcb0f73e55f0a938fe7dcdfcc5ba2ae5` |
| `models/candidate/v6_action_ablation_001/flat_sampler_only_loso_gZ-36.pt` | yes | 1,103,233 | `47c9334c6d9bf0215a7212cd09175c719780484b3b6d3ca08dce8a0bed5c0a65` |
| `models/candidate/v6_action_ablation_001/flat_unweighted_loso_gB-9.pt` | yes | 1,103,167 | `e01a9bf04d8b4088c2d68d2e3de22419d9218fc462058c875df1580fd9143b3e` |
| `models/candidate/v6_action_ablation_001/flat_unweighted_loso_gC-14.pt` | yes | 1,103,189 | `ecb5149a38dcd88846e1d7f078df23e6f7527be1c82a13d4d9e56578d1ed5861` |
| `models/candidate/v6_action_ablation_001/flat_unweighted_loso_gZ-36.pt` | yes | 1,103,189 | `4efac82e5f29cd0d680f61a1bfca861f2e2947c9377ad9d8d9ef76769bf3d567` |
| `models/candidate/v6_action_ablation_001/hierarchical_loss_only_loso_gB-9.pt` | yes | 1,104,453 | `d78e906dbf927cdd37f37231b7e5c81d69c1532b8bccf5b20e100e90f5245b8c` |
| `models/candidate/v6_action_ablation_001/hierarchical_loss_only_loso_gC-14.pt` | yes | 1,104,477 | `a804687c6d2268a88801441e46cc7fa358c27ca15d74086d6bda53828a3ad080` |
| `models/candidate/v6_action_ablation_001/hierarchical_loss_only_loso_gZ-36.pt` | yes | 1,104,477 | `ebb0835a3bec116377e27b5bf8881a4efe43ce7bca4c0b912ab2544e416b78c3` |

The `.pt` candidates are intentionally excluded from Docker context and are not automatically committed. Registry metadata is in `models/model_registry_v6.json`.
