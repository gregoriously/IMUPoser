# %%
from pathlib import Path

import pytorch_lightining as pl
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.datasets.utils import get_datamodule
from imuposer.models.utils import get_model
from imuposer.utils import get_parser

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
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
)

checkpoint_dir = config.checkpoint_path
with open(checkpoint_dir / "best_model.txt") as f:
    best_model = Path(f.readlines()[0].strip()).name
    print(f"[DEBUG] Best model: {best_model}")
model = get_model(config)
model = model.load_from_checkpoint(checkpoint_dir / best_model)

datamodule = get_datamodule(config)
trainer = pl.Trainer()

trainer.test(
    model, datamodule=datamodule
)  # this uses the data loaders defined in dataset/utils.py
# not that this will not work because there is currently no test step in the model definition.
