import os
import numpy as np

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from torchvision.io import read_image
from torch.utils.data import Dataset



class CustomImageDatasetFromCsv(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, label_transform=None):
        self.img_labels = dataframe  # pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        print(img_path)
        image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # if self.label_transform:
        #    label = self.target_transform(label)

        return (image, img_path)

# dataset_path = os.path.join('E:\\Documentos\\Universidad\\4º\\TFG\\lofi-transoformer', 'dataset')
#
# image_path = os.path.join(dataset_path, 'images')
#
# df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
# train_df, val_df = train_test_split(df)
#
# train_data = CustomImageDatasetFromCsv(train_df, image_path, transform=None)
# val_data = CustomImageDatasetFromCsv(val_df, image_path, transform=None)
#
# batch_size = 3
# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# for item_img , item_x in train_dataloader:
#     for item in item_x:
#         print("- "+item)
#         img = Image.open(item)
#         print(img)
#         plt.imshow(img, cmap='magma')
#
#     print("\n-------------------------------------------------------\n")

# sample_images, x = next(iter(train_dataloader))

