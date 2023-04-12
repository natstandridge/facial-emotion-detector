import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import numpy as np
from PIL import Image

'''
FEC2013 contains three columns: 1, emotion number - 2, grayscale pixel values
for 48x48 images - 3, data purpose (training or testing)

Emotion codes:
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

'''

class FER2013(Dataset):
    def __init__(self, train, transform):
        super().__init__()
        self.filepath = 'assets/fer2013.csv'
        self.transform = transform
        self.train = train

        # consider renaming emotions for more logical reading by humans
        self.emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

        with open(self.filepath) as f: # read all the csv using readlines
            self.data = f.readlines()

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        with open(self.filepath) as f:
            emotion, img, usage = f.readlines()[i].split(",") ## column names already removed from dataset
        
        emotion = int(emotion)
        img = img.split(" ") ## split pixel values space-wise
        img = np.array(img, 'int')
        img = np.array(img, dtype=np.uint8)
        img = img.reshape(48, 48) ## pixels are in one line, this reshapes them to the proper size
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return((img, emotion))
