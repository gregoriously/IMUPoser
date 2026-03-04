# IMUPoser TODO

0. Review original paper — verify the code faithfully implements the paper (data pipeline, model arch, training procedure, evaluation metrics)
1. ~~Test script for Global Model — evaluate on dip_test.pt, report per-combo metrics~~ ✓
   1a. Implement test_step in IMUPoserModel — compute MPJRE (degrees) and MPJPE (mm) via FK, log per-combo breakdowns
2. Fine-tune script — load pretrained checkpoint, train on dip_train.pt
3. Test script for fine-tuned model — evaluate on dip_test.pt
4. Config update — add `pretrained_checkpoint` field (default None)
5. get_model update — handle checkpoint loading internally instead of requiring `pretrained` arg
6. Review datasets/ and refactor GlobalModelDataset/FineTuneDIP into a single extensible base class — configurable file paths, sensor count, and combo masking to support fine-tuning on self-gathered data
