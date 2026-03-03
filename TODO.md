# IMUPoser TODO

0. Review original paper — verify the code faithfully implements the paper (data pipeline, model arch, training procedure, evaluation metrics)
1. Test script for Global Model — evaluate on dip_test.pt, report per-combo metrics
2. Fine-tune script — load pretrained checkpoint, train on dip_train.pt
3. Test script for fine-tuned model — evaluate on dip_test.pt
4. Config update — add `pretrained_checkpoint` field (default None)
5. get_model update — handle checkpoint loading internally instead of requiring `pretrained` arg
6. Generic fine-tune dataset class — configurable filenames instead of hardcoded DIP
