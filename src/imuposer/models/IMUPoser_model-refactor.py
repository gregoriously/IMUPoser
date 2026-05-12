from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn

from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.smpl.parametricModel import ParametricModel

# original imported loss functions module but this was unused (used a weighted MSE) check paper

N_JOINTS = 24
N_DEVICES = 5  # previously derived from amass combos in the config, useful when evaluating specific combos, but for model creation specifically for IMUPoser this will always be 5 per the dataloader, and masking approach
SMPL_MODEL_PATH = "/path/"
DEVICE = "cuda"  # check correct
BATCH_SIZE = 32


@dataclass
class ModelConfig:
    n_hidden: int = 512
    r6d: bool = True
    dropout: float = 0.2
    n_lstm_layers: int = 2
    loss: str = "mse"
    lr: float = 3e-4
    use_joint_loss: bool = True
    batch_size: int = BATCH_SIZE  # this will need to match dataloader batchsize, think about where this lives. Model will not crash if incorrect or mismatched, but will not accurately calculate metrics.


class IMUPoserModel(pl.LightningModule):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        n_input = N_DEVICES * 12
        n_hidden = cfg.n_hidden
        n_lstm_layers = cfg.n_lstm_layers
        if cfg.r6d:
            n_output = 6 * N_JOINTS
        else:
            n_output = 9 * N_JOINTS

        # training parameters
        if cfg.loss == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()
        self.use_joint_loss = cfg.use_joint_loss
        if self.use_joint_loss:
            self.bodymodel = ParametricModel(SMPL_MODEL_PATH, DEVICE)

        self.lr = cfg.lr
        self.batch_size = cfg.batch_size

        self.save_hyperparameters()

        # Layers:
        self.dropout_layer = nn.Dropout(cfg.dropout)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            n_hidden,
            n_hidden,
            num_layers=n_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )  # bidirectional baked in.
        self.linear2 = nn.Linear(n_hidden * 2, n_output)

    def forward(self, x, x_lens, h=None):
        x = self.dropout_layer(x)
        x = self.linear1(x)
        x = self.relu(x)
        # packing the padded
        x = nn.utils.rnn.pack_padded_sequence(
            x, x_lens, batch_first=True, enforce_sorted=False
        )
        # LSTM layer
        x, h = self.bilstm(x, h)
        # pad the packed
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.linear2(x)
        pred_pose = x
        return pred_pose

    def _shared_step(self, batch, batch_idx):
        imu_inputs, target_pose, imu_lens, _ = batch  # drop output lens
        pred_pose = self(imu_inputs, imu_lens)
        # pred_pose = _pred[:,:, self.n_pose_output], is the original, not necessary as it can only output the correct num of feature dimensions
        # same for target pose
        loss = self.loss(pred_pose, target_pose)
        if self.use_joint_loss:
            # loss += self._joint_loss(pred_pose, target_pose) #the below is safer on tensors
            loss = loss + self._joint_loss(pred_pose, target_pose)
        return loss, pred_pose, target_pose

    def _joint_loss(self, pred_pose, target_pose):
        pred_joint = self._fk(pred_pose)
        target_joint = self._fk(
            target_pose
        )  # this could be pre-calculated in the dataloader
        joint_pos_loss = self.loss(pred_joint, target_joint)
        return joint_pos_loss

    def _mpjpe(self, pred_pose, target_pose):
        pred_joint = self._fk(pred_pose)
        target_joint = self._fk(target_pose)
        # previously aligned by pelvis, but fk is calculated from the root joint anyway - worth checking.
        mpjpe = torch.norm(pred_joint - target_joint, dim=-1).mean()
        return mpjpe

    def _mpjre(self, pred_pose, target_pose):
        # need to the think about the inertial frame of the calculation....
        R_pred = r6d_to_rotation_matrix(pred_pose)
        R_target = r6d_to_rotation_matrix(target_pose)
        R_diff = R_pred @ R_target.transpose(-1, -2)
        angle = torch.acos(
            torch.clamp(
                (R_diff.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1, 1
            )  # clamp is necessary for floating point
        )
        mpjre = angle.mean() * (180 / torch.pi)
        return mpjre

    def _mpjve(self, pred_pose, target_pose):
        raise NotImplementedError

    def _fk(self, pose):
        pose_joint = self.bodymodel.forward_kinematics(
            pose=r6d_to_rotation_matrix(pose).view(-1, 216)
        )[1]
        return pose_joint

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        loss, pred_pose, target_pose = self._shared_step(batch, batch_idx)
        mpjpe = self._mpjpe(pred_pose, target_pose)
        mpjre = self._mpjre(pred_pose, target_pose)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_mpjpe",
            mpjpe,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_mpjre",
            mpjre,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return {"loss": loss, "mpjpe": mpjpe, "mpjre": mpjre}

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        loss, pred_pose, target_pose = self._shared_step(batch, batch_idx)
        self.log(
            "pred_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return {"loss": loss, "pred": pred_pose, "true": target_pose}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # the below is no longer needed in pytl 2.x, and can be handled by self.log as we dont do anything fancy...
    # this is in the docs https://pytorch-lighting.readthedocs.io/en/latest/common/lightning_module.html

    # def training_epoch_end(self, outputs):
    #     self.epoch_end_callback(outputs, loop_type="train")
    #
    # def validation_epoch_end(self, outputs):
    #     self.epoch_end_callback(outputs, loop_type="val")
    #
    # def test_epoch_end(self, outputs):
    #     mpjpes = []
    #     mpjres = []
    #     for out in outputs:
    #         mpjpes.append(out["mpjpe"])
    #
    #         mpjres.append(out["mpjre"])
    #
    #     mpjpe = torch.mean(torch.Tensor(mpjpes))
    #     mpjre = torch.mean(torch.Tensor(mpjres))
    #     self.log(
    #         "test_mpjpe",
    #         mpjpe,
    #         prog_bar=True,
    #         # batch_size=self.batch_size
    #     )
    #
    #     self.log(
    #         "test_mpjre",
    #         mpjre,
    #         prog_bar=True,
    #         # batch_size=self.batch_size
    #     )
    #     self.epoch_end_callback(outputs, loop_type="test")
    #
    # def epoch_end_callback(self, outputs, loop_type="train"):
    #     loss = []
    #     for output in outputs:
    #         loss.append(output["loss"])
    #
    #     # agg the losses
    #     avg_loss = torch.mean(torch.Tensor(loss))
    #     self.log(
    #         f"{loop_type}_loss",
    #         avg_loss,
    #         prog_bar=True,
    #         # batch_size=self.batch_size
    #     )
    #
