# IMUPoser TODO

0. Review original paper — verify the code faithfully implements the paper (data pipeline, model arch, training procedure, evaluation metrics)
1. Test script for Global Model (`scripts/2. Train/2. Test global.py`)
   - [x] Script structure: config, checkpoint loading, trainer.test() call
   - [ ] Fix import typo (`pytorch_lightining` → `pytorch_lightning`)
   - [ ] Fix checkpoint loading pattern (load_from_checkpoint classmethod usage)
   1a. Implement `test_step` in IMUPoserModel
       - [x] Compute MPJRE (Mean Per Joint Rotation Error) in degrees — `accuracy.py:calc_mpjre`
       - [x] Compute MPJPE (Mean Per Joint Position Error), pelvis-aligned — `accuracy.py:calc_mpjpe`
       - [ ] Decide units: currently meters, paper uses cm — add `* 100` somewhere
       - [ ] Stub for MPJVE (mesh vertex error) — paper's primary metric, needs `calc_mesh=True` in FK
       - [ ] Log per-combo metric breakdowns
       - [x] Update `test_epoch_end` to aggregate MPJRE/MPJPE (not just loss)
       - [ ] Redundant FK in test_step (loss + calc_mpjpe both run FK) — optimise later
2. Fine-tune script — load pretrained checkpoint, train on dip_train.pt
3. Test script for fine-tuned model — evaluate on dip_test.pt
4. Config update — add `pretrained_checkpoint` field (default None)
5. get_model update — handle checkpoint loading internally instead of requiring `pretrained` arg
6. Review datasets/ and refactor GlobalModelDataset/FineTuneDIP into a single extensible base class — configurable file paths, sensor count, and combo masking to support fine-tuning on self-gathered data
