from PIL import Image
from torch.utils.data.dataset import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VG256_handler(Dataset):
    def __init__(self, X, Y, data_path,transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path+'/'+self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]

        message = {
            "img_path": self.data_path+'/'+self.X[index],
            "target": y,
            "img": x,
            "index_num": index
        }
        return message

    def __len__(self):
        return len(self.X)
