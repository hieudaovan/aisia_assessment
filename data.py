from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
remove_img = [
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/cardigans/img_31687202.jpg', 
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/women_sweaters/img_31687202.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/women_sweaters/img_31687287.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/women_sweaters/img_31688584.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/women_sweaters/img_31689106.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/cardigans/img_42333857.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/cardigans/img_42333885.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/women_sweaters/img_42334521.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/women_sweaters/img_42333885.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/cardigans/img_44152414.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/women_sweaters/img_44152414.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/women_sweaters/img_45028984.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/men_trousers/img_7352031.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/men_trousers/img_57886900.jpg', 
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/men_trousers/img_58495490.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/men_trousers/img_59404143.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/men_sweaters/img_46642661.jpg',
    '/space/hotel/hieud/mlflow_aisia/image/AISIA_BOUTIQUE_DATASET/men_sweaters/img_46642684.jpg',
    ]
class ImageData(Dataset):
    def __init__(self, data_pd, transform=None):
        self.data_pd = data_pd
        print(self.data_pd['category_index'].unique())
        self.transform = transform
        #print(data_pd)

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):
        label = self.data_pd.iloc[idx]['category_index']
        img_path = self.data_pd.iloc[idx]['img_path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print(self.data_pd['category_index'].unique())
        #print(img_path)
        if self.transform:
            img = self.transform(img)
        # return img, torch.eye(11)[label]
        return img,label


class ImageDataset():
    def __init__(self, img_dir) -> None:
        self.img_dir = img_dir
        categories = os.listdir(self.img_dir)
        categories.remove(".DS_Store")
        self.class_to_idx = {cat: idx for idx, cat in enumerate(categories) }
        self.data_pd = self.create_data_frame()
        print(img_dir)
        #print(self.class_to_idx)
        
    def create_data_frame(self) -> pd.DataFrame: 
        img_paths, img_cats, img_cats_idx = [], [], []
        categories = os.listdir(self.img_dir)
        categories.remove(".DS_Store")
        for category in categories:
            if category != ".DS_Store":# and category=="tanks":
                category_dir = os.path.join(self.img_dir, category)
                for img_id in os.listdir(category_dir):
                    #print(category_dir+'/'+img_id)
                    if img_id == ".DS_Store" :
                        continue
                    
                   
                    img_path = category_dir+'/'+img_id
                    
                    if img_path in remove_img:
                        continue
                    #print(img_path)
                    img_paths.append(img_path)
                    img_cats_idx.append(self.class_to_idx[category])
                    img_cats.append(category)
        data = {'img_path': img_paths, 'category': img_cats, 'category_index': img_cats_idx}
        data_pd = pd.DataFrame(data)
        # print(data_pd.head(1))
        # one_hot_encoded = pd.get_dummies(data_pd['category'])
        if 'image' not in self.img_dir:
            data_pd.to_csv('img_without_woman_coats.csv', index=False)  
        return data_pd

    def get_dataloader(self, batch_size=16, num_workers=1,  transform=None):
        # print(len(self.data_pd))
        #print(self.data_pd['category_index'].unique())
        train_val_pd , test_pd = train_test_split(self.data_pd, test_size=0.1, random_state=42, stratify=self.data_pd['category'])
        train_pd , val_pd = train_test_split(train_val_pd, test_size=0.2, random_state=42, stratify=train_val_pd['category'])
        # train_pd.to_csv('train.csv', index=False)
        # val_pd.to_csv('val.csv', index=False)
        # test_pd.to_csv('test.csv', index=False)
        train_dataset, val_dataset, test_dataset = ImageData(train_pd,transform), ImageData(val_pd,transform), ImageData(test_pd,transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        return train_dataloader, val_dataloader,  test_dataloader

# test = ImageDataset('./image/AISIA_BOUTIQUE_DATASET')
# train_dataloader, val_dataloader,  test_dataloader = test.get_dataloader(4)
# for x, y in train_dataloader:
#     print(x.shape, y.shape, len(train_dataloader)*4)
#     break


