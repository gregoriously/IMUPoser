# %%
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
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

# instantiate the old model GlobalModelIMUPoser
# load its checkpoint
# assign it to pretrained, get_model(config, pretrained) as an argument
model = get_model(config)
