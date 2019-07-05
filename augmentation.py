import random
from PIL import Image
import numpy as np
import torchvision.datasets as vdatasets


class SamplePairing(object):
    def __init__(self, train_data_dir, p=1):
        self.train_data_dir = train_data_dir
        self.pool = self.load_dataset()
        self.p = p

    def load_dataset(self):
        dataset_train_raw = vdatasets.ImageFolder(self.train_data_dir)
        return dataset_train_raw

    def __call__(self, img):
        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            img_a_array = np.asarray(img)

            # pick one image from the pool
            img_b, _ = random.choice(self.pool)
            img_b_array = np.asarray(img_b.resize((197, 197)))

            # mix two images
            mean_img = np.mean([img_a_array, img_b_array], axis=0)
            img = Image.fromarray(np.uint8(mean_img))

        return img
