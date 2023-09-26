from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from libauc.datasets import CheXpert

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# paramaters
SEED = 123
BATCH_SIZE = 128
lr = 1e-4
weight_decay = 1e-5

# dataloader
root = './CheXpert-v1.0'
# Index: -1 denotes multi-label mode including 5 diseases
traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1)
testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

# model
set_all_seeds(SEED)
model = DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=5)
model = model.cuda()

# define loss & optimizer
CELoss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# training
best_val_auc = 0 
for epoch in range(10):
    for idx, data in enumerate(trainloader):
        train_data, train_labels = data
        train_data, train_labels  = train_data.cuda(), train_labels.cuda()
        y_pred = model(train_data)
        loss = CELoss(y_pred, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # validation  
        if idx % 100 == 0:
            model.eval()
            with torch.no_grad():    
                test_pred = []
                test_true = [] 
                for jdx, data in enumerate(testloader):
                    test_data, test_labels = data
                    test_data = test_data.cuda()
                    y_pred = model(test_data)
                    test_pred.append(y_pred.cpu().detach().numpy())
                    test_true.append(test_labels.numpy())
            
                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                val_auc_mean =  roc_auc_score(test_true, test_pred) 
                model.train()

                if best_val_auc < val_auc_mean:
                    best_val_auc = val_auc_mean
                    torch.save(model.state_dict(), 'ce_pretrained_model.pth')

                print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc ))