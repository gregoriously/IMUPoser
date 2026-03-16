# %%
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from imuposer.config import Config, amass_combos
from imuposer.datasets.utils import get_datamodule
from imuposer.models.utils import get_model
from imuposer.utils import get_parser

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
# TODO: add --pretrained_checkpoint_dir arg to get_parser instead of requiring manual path
parser.add_argument(
    "--pretrained_checkpoint_dir",
    required=True,
    help="Path to the pretrained global model checkpoint directory (contains best_model.txt)",
)
args = parser.parse_args()
combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment

# %%
# Load pretrained global model
# TODO: this is janky — we create a throwaway Config just to get the SMPL paths and model shape,
# then manually load the checkpoint. Ideally Config would accept a pretrained_checkpoint path
# and get_model would handle loading internally (see TODOs 4 & 5).
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
    mkdir=False,  # don't create a new checkpoint dir for loading
)

pretrained_checkpoint_dir = Path(args.pretrained_checkpoint_dir)
with open(pretrained_checkpoint_dir / "best_model.txt") as f:
    best_model = Path(f.readlines()[0].strip()).name
    print(f"Pretrained model: {best_model}")

pretrained_model = get_model(pretrained_config).load_from_checkpoint(
    pretrained_checkpoint_dir / best_model, config=pretrained_config
)

# %%
# Create fine-tune model wrapping the pretrained one
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
)

model = get_model(config, pretrained_model)
print("Fine-tune model created")

datamodule = get_datamodule(config)
print("Datamodule loaded")

checkpoint_path = config.checkpoint_path

# %%
wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

early_stopping_callback = EarlyStopping(
    monitor="validation_step_loss",
    mode="min",
    verbose=False,
    min_delta=0.00001,
    patience=5,
)
checkpoint_callback = ModelCheckpoint(
    monitor="validation_step_loss",
    mode="min",
    verbose=False,
    save_top_k=5,
    dirpath=checkpoint_path,
    save_weights_only=True,
    filename="epoch={epoch}-val_loss={validation_step_loss:.5f}",
)

trainer = pl.Trainer(
    fast_dev_run=fast_dev_run,
    logger=wandb_logger,
    max_epochs=1000,
    accelerator="gpu",
    devices=[0],
    callbacks=[early_stopping_callback, checkpoint_callback],
    deterministic=True,
)

# %%
print("Begin fine-tuning")
trainer.fit(model, datamodule=datamodule)

# %%
with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(
        f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}"
    )
print(f"Best model saved to: {checkpoint_callback.best_model_path}")
