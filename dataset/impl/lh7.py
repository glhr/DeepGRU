import os
from pathlib import Path
import numpy as np

from DeepGRU.dataset.dataset import Dataset, HyperParameterSet
from DeepGRU.dataset.augmentation import AugRandomScale, AugRandomTranslation
from DeepGRU.dataset.impl.lowlevel import Sample, LowLevelDataset
from DeepGRU.utils.logger import log

import random
random.seed(1570254494)

from DeepGRU.utils.utils import get_path_from_root

# ----------------------------------------------------------------------------------------------------------------------
class DatasetLH7(Dataset):
    def __init__(self, root=get_path_from_root("data/LH7"), num_synth=0):
        super(DatasetLH7, self).__init__("LH7", root, num_synth)

    def _load_underlying_dataset(self):
        self.underlying_dataset = self._load_lh7_interaction(verbose=False)
        self.num_features = 27  # 3D world coordinates joints of 1 person (9 joints x 3 dimensions).
                                # Each row is one frame.
        self.num_folds = 3     # This dataset has 3 folds

    def get_hyperparameter_set(self):
        return HyperParameterSet(learning_rate=0.001,
                                 batch_size=64,
                                 weight_decay=0,
                                 num_epochs=100)

    def _get_augmenters(self, random_seed):
        return [
            AugRandomScale(3, self.num_synth, random_seed, 0.7, 1.3),
            AugRandomTranslation(3, self.num_synth, random_seed, -1, 1),
        ]

    def _load_lh7_interaction(self, unnormalize=False, verbose=False):
        """
        Loads the LH7 dataset.
        """

        # Make sure the dataset exists
        self._check_dataset()

        # Each action's label
        LABELS = {"1": "handover",
                  "2": "working",
                  "3": "distracted",
                  "4": "waving"}

        LABELS_INV = dict((v, k) for k, v in LABELS.items())

        # Number of folds
        FOLD_CNT = 3

        # Number of joints
        JOINT_CNT = 9

        # Using 5-fold cross validation as the predefined test
        # (e.g. train[0] test[0] mean test on FOLD[0], train on everything else)
        train_indices = [[] for i in range(FOLD_CNT)]
        test_indices = [[] for i in range(FOLD_CNT)]
        samples = []

        for fname in Path(self.root).glob('**/*.txt'):
            fname = str(fname)

            # Determine sample properties
            subject, label, example = fname.replace(self.root, '').split('/')[-1].split("-")[0:3]
            if label.startswith("wav"):
                label = "waving"
            frame = fname.replace(self.root, '').split('/')[-1].split("-")[-1].split(".")[0]
            label = LABELS_INV[label]

            if verbose:
                log("load: {}, label {}, subject {}, example {}".format(fname,
                                                                        label,
                                                                        subject,
                                                                        example))

            # Now read the actual file
            with open(fname) as f:
                lines = f.read().splitlines()

            pts = DatasetLH7._pnts_from_frames(lines, JOINT_CNT)

            # Make a sample
            samples += [Sample(pts, LABELS[label], subject, fname)]

            # Add the index to train/test indices for each fold
            s_idx = len(samples) - 1

            for fold_idx in range(FOLD_CNT):

                if not random.randint(0,4):
                    # Add the instance as a TESTING instance to this fold
                    test_indices[fold_idx] += [s_idx]

                else:
                    # For all other folds, this guy would be a TRAINING instance
                    train_indices[fold_idx] += [s_idx]

        # k-fold sanity check
        for fold_idx in range(FOLD_CNT):
            assert len(train_indices[fold_idx]) + len(test_indices[fold_idx]) == len(samples)
            # Ensure there is no intersection between training/test indices
            assert len(set(train_indices[fold_idx]).intersection(test_indices[fold_idx])) == 0

        return LowLevelDataset(samples, train_indices, test_indices)

    def _check_dataset(self):
        if not os.path.isdir(self.root):
            log("Dataset files do not exist :(")

    @staticmethod
    def _calculate_motion(np_pts, num_joints):
        """
        Measures the motion of a sample
        """
        total_motion = 0
        motion_per_joint = []

        # Slice along all frames for each joint, then calculate the Euclidean length of that sequence
        for j in range(num_joints):
            joint_col_start = 3 * j
            joint_col_end = 3 * (j + 1)

            pts = np_pts[:, joint_col_start:joint_col_end]

            displacement = DatasetLH7._series_len(pts)
            total_motion += displacement
            motion_per_joint += [displacement]

        assert len(motion_per_joint) == num_joints

        return total_motion, motion_per_joint

    @staticmethod
    def _series_len(pts):
        """
        Computes the path length of a sample
        """
        ret = 0.0

        for idx in range(1, len(pts)):
            ret += np.linalg.norm(pts[idx] - pts[idx - 1])

        return ret

    @staticmethod
    def _pnts_from_frames(lines,JOINT_CNT=9):
        pts = []
        body_pts = {0: []}  # body index -> list of points in the entire sequence
        framecount = len(lines)

        for idx, line in enumerate(lines):
            if isinstance(line, str):
                line = line.split(',')

            body = 0
            pt = np.zeros(3 * JOINT_CNT, dtype=np.float32)

            # Read the (x, y, z) position of the joint of each person (2 people in each frame)
            for i in range(JOINT_CNT):
                pt[3 * i + 0] = float(line[(3 * body * JOINT_CNT) + (3 * i + 0)])
                pt[3 * i + 1] = float(line[(3 * body * JOINT_CNT) + (3 * i + 1)])
                pt[3 * i + 2] = float(line[(3 * body * JOINT_CNT) + (3 * i + 2)])

            body_pts[body] += [pt]

        # Sanity check
        for b_idx in range(1):
            assert len(body_pts[b_idx]) == framecount

        # Sort bodies based on activity (high-activity first)
        bodies_by_activity = sorted(body_pts.items(),
                                    key=lambda item: DatasetLH7._calculate_motion(np.asarray(item[1]),
                                                                                        JOINT_CNT), reverse=True)

        for f in range(framecount):
            for b_idx, bodys_frames in bodies_by_activity:
                pt = bodys_frames[f]
                pts += [pt]

        return pts
