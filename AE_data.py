
from torchvision import transforms
from PIL import ImageFile
import pickle
from glob import glob
import os.path as osp
from PIL import Image, ImageOps
import os
import numpy as np
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True
pil_transform = transforms.ToPILImage(mode='RGB')

def get_all_averages():
    data = {}
    cities = ["austin", "miami", "pittsburgh", "dearborn", "washington-dc", "palo-alto"]
    for city in cities:
        data[city] = calculate_averages(mode="train", city=city)
    pickle.dump(data, open("./AE_normalization.p", "wb"))


def calculate_averages(mode, city):
    dataset = AE_data(mode=mode, city=city, get_mean=True)
    batch_size = 128
    n_workers = 10
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=False
    )

    mean = torch.zeros(1)
    std = torch.zeros(1)
    nb_samples = 0.
    for i, (data, path, city) in enumerate(tqdm.tqdm(loader)):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

        if nb_samples % 5000 == 0:
            print(nb_samples, "Mean:", mean/nb_samples, "Std dev:", std/nb_samples)

    mean /= nb_samples
    std /= nb_samples

    print("Mean:", mean, "Std dev:", std)
    return [mean.item(), std.item()]

class AE_data(Dataset):
    def __init__(self, mode, city, img_size=224, get_mean=False):

        # get the data:
        cities = ["austin", "miami", "pittsburgh", "dearborn", "washington-dc", "palo-alto"]

        self.data = []
        self.cities = []
        if mode == "train":
            for c in cities:
                if c != city:
                    files = glob("../intersections/train/" + c + "/*")
                    self.data.extend(files)
                    self.cities.extend([c]*len(files))

        elif mode == "val":
            for c in cities:
                if c != city:
                    files = glob("../intersections/val/" + c + "/*")
                    self.data.extend(files)
                    self.cities.extend([c] * len(files))
        else:
            files = glob("../intersections/train/" + city + "/*")
            self.data.extend(files)
            self.cities.extend([city] * len(files))


        if get_mean:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])
        else:
            if not osp.exists("./AE_normalization.p"):
                print("Run normalization")
                get_all_averages()
            norm_values = pickle.load(open("./AE_normalization.p", "rb"))


            # populate with normalized values (mean and std):
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[norm_values[city][0]],
                                     std=[norm_values[city][1]])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        city = self.cities[index]
        image = self.transform(ImageOps.invert(Image.open(path).convert('L')))
        return image, path, city

if __name__=="__main__":
    get_all_averages()
    #calculate_averages(mode="train", city="austin")