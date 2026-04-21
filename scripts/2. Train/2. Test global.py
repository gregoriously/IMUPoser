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
parser.add_argument(
    "--checkpoint_dir",
    required=True,
    help="Path to the global model checkpoint directory (contains best_model.txt)",
)
args = parser.parse_args()
combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment

# %%
config = Config(
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

# load best checkpoint
checkpoint_dir = Path(args.checkpoint_dir)
with open(checkpoint_dir / "best_model.txt") as f:
    best_model = Path(f.readlines()[0].strip()).name
    print(f"Best model: {best_model}")

model = get_model(config).load_from_checkpoint(
    checkpoint_dir / best_model, config=config
)

datamodule = get_datamodule(config)

trainer = pl.Trainer(
    fast_dev_run=fast_dev_run,
    accelerator="gpu",
    devices=[0],
)

results = trainer.test(model, datamodule=datamodule)
print(f"\n=== Results for combo: {combo_id} ===")
for k, v in results[0].items():
    print(f"  {k}: {v:.4f}")
