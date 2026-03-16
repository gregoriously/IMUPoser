# %%
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.datasets.utils import get_datamodule
from imuposer.models.utils import get_model
from imuposer.utils import get_parser

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
# TODO: move these to get_parser centrally (see TODO 4/5)
parser.add_argument(
    "--pretrained_checkpoint_dir",
    required=True,
    help="Path to the pretrained global model checkpoint directory (contains best_model.txt)",
)
parser.add_argument(
    "--finetune_checkpoint_dir",
    required=True,
    help="Path to the fine-tuned model checkpoint directory (contains best_model.txt)",
)
args = parser.parse_args()
combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment

# %%
# Step 1: Reconstruct the pretrained global model from its checkpoint.
# TODO: this two-step loading is janky — needed because IMUPoserModelFineTune.__init__
# requires pretrained_model as an arg, and the fine-tune checkpoint doesn't store it
# (save_hyperparameters ignores it). TODOs 4 & 5 would clean this up.
pretrained_config = Config(
    experiment=f"{_experiment}_{combo_id}",
    model="GlobalModelIMUPoser",
    project_root_dir="../../",
    joints_set=amass_combos[combo_id],
    normalize="no_translation",
    r6d=True,
    loss_type="mse",
    use_joint_loss=True,
    device="0",
    mkdir=False,
)

pretrained_checkpoint_dir = Path(args.pretrained_checkpoint_dir)
with open(pretrained_checkpoint_dir / "best_model.txt") as f:
    best_pretrained = Path(f.readlines()[0].strip()).name
    print(f"Pretrained model: {best_pretrained}")

pretrained_model = get_model(pretrained_config).load_from_checkpoint(
    pretrained_checkpoint_dir / best_pretrained, config=pretrained_config
)

# %%
# Step 2: Load fine-tuned model from its checkpoint, passing in the pretrained model
config = Config(
    experiment=f"{_experiment}_finetune_{combo_id}",
    model="GlobalModelIMUPoserFineTuneDIP",
    project_root_dir="../../",
    joints_set=amass_combos[combo_id],
    normalize="no_translation",
    r6d=True,
    loss_type="mse",
    use_joint_loss=True,
    device="0",
    mkdir=False,
)

finetune_checkpoint_dir = Path(args.finetune_checkpoint_dir)
with open(finetune_checkpoint_dir / "best_model.txt") as f:
    best_finetune = Path(f.readlines()[0].strip()).name
    print(f"Fine-tuned model: {best_finetune}")

model = get_model(config, pretrained_model).load_from_checkpoint(
    finetune_checkpoint_dir / best_finetune,
    config=config,
    pretrained_model=pretrained_model,
)

# %%
datamodule = get_datamodule(config)

trainer = pl.Trainer(
    fast_dev_run=fast_dev_run,
    accelerator="gpu",
    devices=[0],
)

results = trainer.test(model, datamodule=datamodule)
print(f"\n=== Fine-tune results for combo: {combo_id} ===")
for k, v in results[0].items():
    print(f"  {k}: {v:.4f}")
