#!/bin/bash
set -e

COMBO="global"
EXPERIMENT="IMUPoserGlobalModel"

echo "=== 1. Train Global Model ==="
python "1. Train Global Model.py" \
  --combo_id "$COMBO" \
  --experiment "$EXPERIMENT"

# Find the most recently created checkpoint dir for this combo
GLOBAL_CKPT=$(ls -dt ../../checkpoints/${EXPERIMENT}_${COMBO}-* | head -1)
echo "Global checkpoint dir: $GLOBAL_CKPT"

echo "=== 2. Test Global Model ==="
python "2. Test global.py" \
  --combo_id "$COMBO" \
  --experiment "$EXPERIMENT" \
  --checkpoint_dir "$GLOBAL_CKPT"

echo "=== 3. Fine-tune on DIP ==="
python "3. Fine tune DIP.py" \
  --combo_id "$COMBO" \
  --experiment "$EXPERIMENT" \
  --pretrained_checkpoint_dir "$GLOBAL_CKPT"

# Find the most recently created fine-tune checkpoint dir
FINETUNE_CKPT=$(ls -dt ../../checkpoints/${EXPERIMENT}_finetune_${COMBO}-* | head -1)
echo "Fine-tune checkpoint dir: $FINETUNE_CKPT"

echo "=== 4. Test Fine-tuned Model ==="
python "4. Test fine tune.py" \
  --combo_id "$COMBO" \
  --experiment "$EXPERIMENT" \
  --pretrained_checkpoint_dir "$GLOBAL_CKPT" \
  --finetune_checkpoint_dir "$FINETUNE_CKPT"

echo "=== Pipeline complete ==="
