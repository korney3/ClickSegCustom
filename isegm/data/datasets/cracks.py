from .grabcut import GrabCutDataset

from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample



class CrackDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='images', masks_dir_name='masks', split="Test",
                 **kwargs):
        super(CrackDataset, self).__init__(**kwargs)
        self.name = 'CrackDataset'
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / split / images_dir_name
        self._insts_path = self.dataset_path / split / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem.split(".")[0]: x for x in self._insts_path.glob('*.*')}



    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]

        prefix = image_name.split(".")[0]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[prefix])
        # mask_path = str(self._masks_paths[f"{prefix}_gt_invGray"])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask <= 128] = -1
        instances_mask[instances_mask > 128] = 1
        # resized = cv2.resize(image, (instances_mask.shape[1], instances_mask.shape[0]))

        return DSample(image, instances_mask, objects_ids=[1], ignore_ids=[-1], sample_id=index)
