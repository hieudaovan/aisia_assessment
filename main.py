import torch
from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy
import mlflow
from torchvision import transforms
import numpy as np
import random
from torchvision.models import resnet50, efficientnet_v2_m,resnext50_32x4d
#=================================================

from data import ImageDataset
from utils import train, evaluate, convert_img2feat, evaluate_performnace
from model import ImageClassifier
device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

# Set seed for Python's built-in random module
random.seed(42)

# Set seed for PyTorch CPU
torch.manual_seed(42)

# Set seed for PyTorch GPU (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
#dataset
dataset = ImageDataset("./AISIA_BOUTIQUE_DATASET")
transform = transforms.Compose([
    #transforms.Resize((210, 210)),
    transforms.ToTensor(),
])

# backbone = nn.Sequential(*list(resnet50(weights="DEFAULT").children())[:9])
# for layer in list(backbone.children())[:7]:
#     for param in layer.parameters():
#         param.requires_grad = False

# backbone = nn.Sequential(*list(efficientnet_v2_m(weights="DEFAULT").children())[:2])
# for layer in list(backbone.children())[0][:-2]:
#     # print(layer)
#     # print("="*90)
#     for param in layer.parameters():
#         param.requires_grad = False
weight_decay =0.01
backbone = nn.Sequential(*list(resnext50_32x4d(weights="DEFAULT").children())[:-1])
# for i , layer in enumerate(list(backbone.children())[:-1]):
#     #print(layer)
#     #print(f"layer{i}"+"="*90)
#     for param in layer.parameters():
#         param.requires_grad = False

#weight_decay = 0.003
model = ImageClassifier(backbone).to(device)
learning_rate = 1e-5
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()

#---------------------------------------------------
#hypper param
epochs = 30
batch_size= 32

#----------------------------------------------------

train_dataloader, val_dataloader, test_dataloader, train_val_pd_dataloader = dataset.get_dataloader(batch_size=batch_size, num_workers=8, transform=transform)

mlflow.set_tracking_uri("file:///space/hotel/hieud/mlflow_aisia/mlruns")

# Create a new MLflow Experiment
mlflow.set_experiment("eval_time")

description= f"""
train longer
"""

with mlflow.start_run(run_name=description, description=description) as run:
    params = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "Adam",
        "weight_decay":weight_decay,
    }
    # Log training parameters.
    mlflow.log_params(params)

    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("./model_summary.txt")
    print(dataset.data_pd['category_index'].unique())

    for t in range(epochs+1):
        print(f"Epoch {t}\n-------------------------------")
        train(train_val_pd_dataloader, model, loss_fn, metric_fn, optimizer, t, device)
        evaluate(test_dataloader, model, loss_fn, metric_fn, t, device)
    convert_img2feat( model.cpu(), 'img_without_woman_coats.csv')

    # precision = evaluate_performnace('val.csv')
    # mlflow.log_metric("Precision", f"{precision:2f}")
    mlflow.pytorch.log_model(model, "model")
    print(description)