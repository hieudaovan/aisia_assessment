import torch
import mlflow
from tqdm import tqdm
# from data import ImageData
import pandas as pd
# from torch.utils.data import DataLoader
import cv2
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import os
import json
import time
from torchvision import transforms
import matplotlib.pyplot as plt
progressive_transform = transforms.Compose(
    [
#     transforms.Resize((210, 210)),

        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ColorJitter(brightness=.5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3    ),
#        # transforms.RandomVerticalFlip(p=(0.3)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
# mlflow.log_params(progressive_transform)
device = "cuda" if torch.cuda.is_available() else "cpu"
def train(dataloader, model, loss_fn, metrics_fn, optimizer, epoch, device):
    """Train the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
        epoch: an integer, the current epoch number.
    """
 

    model.train()
    train_loss, train_accuracy = 0, 0
    labels, preds =[],[]
    for  i, (X, y) in enumerate(tqdm(dataloader)):
        X = progressive_transform(X)
        
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        if epoch % 5 ==0 and i==0:
            file_name = f" epoch:{epoch}, batch_size{i}"
            #show_images(X, file_name, y, pred)
        #print(pred.shape[1], y.shape)
        
        loss = loss_fn(pred, y)
        accuracy = metrics_fn(pred, y)
        labels += (y.tolist())
        preds +=torch.argmax(pred.cpu(), dim=1).tolist()
        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mlflow.log_metric("loss/batch", f"{loss.item():2f}", step=(epoch*len(dataloader)+i))
        train_loss += loss.item() 
        train_accuracy += accuracy.item()

    if epoch % 5 ==0:
        name = f"./plot_train_pred/predtion_train_epoch : {epoch}"
        plot_pred_distribution(labels, preds, name)
    num_batches = len(dataloader)
    train_loss /= num_batches
    train_accuracy /= num_batches
    mlflow.log_metric("train_loss", f"{train_loss:2f}", step=epoch)
    mlflow.log_metric("train_acc", f"{train_accuracy:2f}", step=epoch)
    #print(set(labels), set(preds), )
    print(f"train_loss: {train_loss:2f} train_accuracy: {train_accuracy:2f}")
    
def evaluate(dataloader, model, loss_fn, metrics_fn, epoch, device):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        epoch: an integer, the current epoch number.
    """
    labels, preds = [],[]
    num_batches = len(dataloader)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        for  (X, y) in tqdm(dataloader):
            X = val_transform(X)
            labels += (y.tolist())
            X, y = X.to(device), y.to(device)
            pred = model(X)
            #preds +=torch.argmax(pred.cpu(), dim=1).tolist()
            #print(pred.shape, y.shape)
            eval_loss += loss_fn(pred, y).item()
            eval_accuracy += metrics_fn(pred, y).item()
            preds +=torch.argmax(pred.cpu(), dim=1).tolist()
    eval_loss /= num_batches
    eval_accuracy /= num_batches
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch)
    
    if epoch % 5 ==0:
        name = f"./plot_val_pred/predtion_val_epoch : {epoch}"
        plot_pred_distribution(labels, preds, name)
    print(f"eval_loss: {eval_loss:2f} eval_accuracy: {eval_accuracy:2f}")

def convert_img2feat(model, data_path):
    #data_pd = pd.read_csv('/space/hotel/hieud/mlflow_aisia/img.csv')
    data_pd = pd.read_csv('/space/hotel/hieud/mlflow_aisia/img_without_woman_coats.csv')
    train_val_pd , test_pd = train_test_split(data_pd, test_size=0.1, random_state=42, stratify=data_pd['category'])
    train_pd , val_pd = train_test_split(train_val_pd, test_size=0.2, random_state=42, stratify=train_val_pd['category'])
    latent_dict = {} 
    for i in tqdm(range(len(test_pd))):
        row = test_pd.iloc[i]
        image = cv2.imread(row['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image,  dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            # x =
            x = model(image_tensor).cpu()
            latent_feature = x.tolist()
        latent_dict[row['img_path']] = latent_feature

    with open(f'/space/hotel/hieud/mlflow_aisia/latent_features_deep_learning_test.json', 'w') as json_file:
        json.dump(latent_dict, json_file)

def similar_diff(img1_feature: list, img2_feature:list):
    diff = np.array(img1_feature)-np.array(img2_feature)
    return np.linalg.norm(diff)

def search(query_image_path):
    with open('/space/hotel/hieud/mlflow_aisia/latent_features_val.json', 'r') as json_file:
    # Load the JSON data into a Python dictionary
        laten_features = json.load(json_file)
    diff_list = {}
    # start = time.time()
    for i in laten_features.keys():
        diff_list[i] = similar_diff(laten_features[query_image_path], laten_features[i])
    # print(time.time()-start)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     diff_list = {i: result for i, result in zip(laten_features.keys(), executor.map(lambda key: similar_diff(laten_features[key], laten_features[query_image_path]), laten_features.keys()))}
    # start = time.time()
    top_5_recommend = sorted(diff_list.items(), key=lambda x: x[1], reverse=False)[:5]
    # print(time.time()-start)
    # print("="*90)
    return top_5_recommend

def precision_at_K(query_image_path, top_5_recommend):
    y_cat = np.array([query_image_path.split('/')[-2]]*5)
    # y_cat = np.array(['cardigans']*5)
    y_pred = np.array([cat[0].split('/')[-2] for cat in top_5_recommend])
    comparison_array = (y_cat == y_pred)
    true_positives = np.sum(comparison_array)

    return true_positives/(true_positives+  len(comparison_array)-true_positives)


# def create_data_csv(img_dir):
#     class_to_idx = {cat: idx for idx, cat in enumerate(os.listdir(img_dir)) if cat != ".DS_Store"}
#     img_paths, img_cats, img_cats_idx = [], [], []
#     for category in os.listdir(img_dir):
#         if category != ".DS_Store":# and category=="tanks":
#             category_dir = os.path.join(img_dir, category)
#             for img_id in os.listdir(category_dir):
#                 #print(category_dir+'/'+img_id)
#                 if img_id == ".DS_Store" :
#                     continue
#                 img_path = category_dir+'/'+img_id
#                 #print(img_path)
#                 img_paths.append(img_path)
#                 img_cats_idx.append(class_to_idx[category])
#                 img_cats.append(category)
#     data = {'img_path': img_paths, 'category': img_cats, 'category_index': img_cats_idx}
#     data_pd = pd.DataFrame(data)
#     # print(data_pd.head(1))
#     # one_hot_encoded = pd.get_dummies(data_pd['category'])
#     #data_pd.to_csv('img.csv', index=False)  
#     return data_pd

def evaluate_performnace(csv_path):
    test_pd = pd.read_csv(csv_path)
    total_precision = 0
    for i in tqdm(range(len(test_pd))):
        image_path = test_pd.iloc[i]['img_path']
        top_5 = search(image_path)
        precision = precision_at_K(image_path, top_5)
        total_precision += precision
    total_precision /= (len(test_pd))
    return total_precision

def show_images(images, name, labels, preds):
    """Show images in a grid."""
    class_dict = {
        'cardigans': 0, 
        'men_trousers': 1, 
        'women_sweaters': 2, 
        'polo_shirts': 3, 
        'women_coats': 4, 
        'women_pants': 5, 
        'tanks': 6, 
        'dresses': 7, 
        'women_tshirts': 8, 
        'men_sweaters': 9, 
        'men_coats': 10}
    images = images.cpu()
    labels = labels.cpu()
    preds =  torch.argmax(preds.cpu(), dim=1)
    
    plt.figure(figsize=(30, 20))
    for i, image in enumerate(images):
        plt.subplot(4, 8, i+1)
        plt.imshow(image.permute(1, 2, 0))
        #print(f"{labels[i], preds[i]}")
        plt.title(f"{get_key_by_value(class_dict,labels[i]), get_key_by_value(class_dict,preds[i])}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"./plot_X/{name}")
    plt.close()

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def plot_pred_distribution(labels, preds,name):
    # Plot distribution of labels
   # max_count = max(max(true_counts), max(pred_counts))

    #print(len(labels),(preds))
    num_classes = 11 
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    labels, true_counts = np.unique(np.array(labels), return_counts=True)
    preds, pred_counts = np.unique(np.array(preds), return_counts=True)
    max_count = max(max(true_counts), max(pred_counts))

    plt.bar(labels, true_counts, align='center')
    #Zzplt.gca().set_xticks(labels)
    plt.xlabel('True Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of True Labels')
    plt.xticks(range(num_classes))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max_count)
    # Plot distribution of predictions
    plt.subplot(1, 2, 2)
    #preds, counts = np.unique(np.array(preds), return_counts=True)
    plt.bar(preds, pred_counts, align='center')
    #plt.gca().set_xticks(preds)
    plt.xlabel('Predicted Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions')
    plt.xticks(range(num_classes))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max_count)
    plt.tight_layout()
    plt.savefig(f"{name}")
    plt.close()