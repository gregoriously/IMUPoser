from abc import ABC, abstractmethod

# window size should be hard coded, as the paper provides a value of exactly 5s, whereas the original code calculates
from pathlib import Path  # should be move to config as we only use to define paths.

import torch
from torch.utils.data import Dataset

from imuposer import math
from imuposer.config import amass_combos

WINDOW_LENGTH = 5  # move to imuposer.config.py
DATA_FPS = 25
WINDOW_FRAMES = int(
    WINDOW_LENGTH * DATA_FPS
)  # 125, originally hard coded config values that evaluated to 300 * 25 // 60 = 125

ACC_SCALE = 30.0  # move to imuposer.config.py
PREPROCESSED_DATA_DIR = Path("/path/")  # move to imuposer.config.py
TRAIN_VAL_SPLIT = 0.9
R6D_OUTPUT = True  # this is configurable in the main repo, but for the paper the output is SMPL thetas as 6d rotations (N = 144), per sect 4.1.
# Note, need to check that the input poses definitely are as rot mats. (I thought they were axis angle for all.)
PRED_JOINTS_SET = list(
    range(24)
)  # again, configurable in original repo, dont know why you would want to change? as this is how many SMPL joints there are


class IMUPoserDatasetBase(Dataset, ABC):
    def __init__(self, split="train", combo_to_test="all"):
        self.imu = []
        self.pose = []
        self.split = split
        self.combo_to_test = combo_to_test
        self.data_files = self.get_data_files(self.split)
        self.load_data()

    @abstractmethod
    def get_data_files(
        self, split
    ) -> list:  # why do this as an abstract method when we can pass the data_files as argument to the class? probably because AMASS is large, and we can keep the AMASS dataset class kept out of the way and define it in this method. Will have a think about this though.
        ...

    def build_imu(self, acc, ori, fpose):
        if self.combo_to_test == "all":
            self.build_imu_all_combos(acc, ori, fpose)
        elif self.combo_to_test in amass_combos:
            if self.split == "train":
                raise ValueError("Training must be done with all device combos")
            else:
                self.build_imu_single_combo(self.combo_to_test, acc, ori, fpose)
        else:
            raise ValueError("combo not in amass_combos")

    def build_imu_single_combo(self, combo, acc, ori, fpose):
        """
        Allows testing of a single combo
        """

        _combo_acc = torch.zeros_like(acc)
        _combo_ori = torch.zeros((3, 3)).repeat(
            ori.shape[0], 5, 1, 1
        )  # could this be torch.zeroes_like(ori)?

        _combo_acc[:, amass_combos[combo]] = acc[:, amass_combos[combo]]
        _combo_ori[:, amass_combos[combo]] = ori[:, amass_combos[combo]]

        imu_inputs = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1)

        self.imu.extend(torch.split(imu_inputs, WINDOW_FRAMES))
        self.pose.extend(torch.split(fpose, WINDOW_FRAMES))

    def build_imu_all_combos(
        self, acc, ori, fpose
    ):  # should this belong in the abstract class, or only in a multicombo one? probs here, this will always apply to train
        for _combo in amass_combos:
            self.build_imu_single_combo(_combo, acc, ori, fpose)

    def load_data(self):
        for fname in self.data_files:
            fdata = torch.load(PREPROCESSED_DATA_DIR / fname)

            for i in range(len(fdata["acc"])):
                # inputs
                facc = fdata["acc"][i]
                fori = fdata["ori"][i]

                # load all the data, dropping index 5 which is belly of DIP, pelvis of AMASS. Why generate it in the preprocessing if it isnt used and gets dropped?
                # Note, I believe there is a bug in the preprocessing, selecting the elbows over the wrists. This will need checking but unsure how.
                glb_acc = facc.view(-1, 6, 3)[:, [0, 1, 2, 3, 4]] / ACC_SCALE
                glb_ori = fori.view(-1, 6, 3, 3)[:, [0, 1, 2, 3, 4]]

                acc = glb_acc
                ori = glb_ori

                # outputs
                fpose = fdata["pose"][i]
                fpose = fpose.reshape(fpose.shape[0], -1)

                # here the original code did a window calculation step
                # maybe add an assertion to check while developing?

                self.build_imu(acc, ori, fpose)

    def __getitem__(self, idx):
        _input = self.imu[idx].float()

        _pose = self.pose[idx].float()
        if R6D_OUTPUT:
            _output = (
                math.rotation_matrix_to_r6d(_pose)
                .reshape(-1, 24, 6)[:, PRED_JOINTS_SET]
                .reshape(-1, 6 * len(PRED_JOINTS_SET))
            )
        else:
            _output = _pose
        return _input, _output

    def __len__(self):
        return len(self.imu)


class AMASSTrainDIPTest(IMUPoserDatasetBase):
    def get_data_files(self, split):
        if split == "train":
            data_files = [
                x.name for x in PREPROCESSED_DATA_DIR.iterdir() if "dip" not in x.name
            ]  # excludes the future DIP fine-tune data, and dip_test. Basically gets all of AMASS
        else:
            data_files = ["dip_test.pt"]
        return data_files


class DIPFineTuneDIPTest(IMUPoserDatasetBase):
    def get_data_files(self, split):
        if split == "train":
            data_files = ["dip_train.pt"]
        else:
            data_files = ["dip_test.pt"]
        return data_files


# Implement an IMUPoser dataset class?
