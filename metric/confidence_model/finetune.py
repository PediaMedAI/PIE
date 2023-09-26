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

# parameters
class_id = 0  # 0:Cardiomegaly, 1:Edema, 2:Consolidation, 3:Atelectasis, 4:Pleural Effusion 
root = './CheXpert-v1.0'

# paramaters
SEED = 123
BATCH_SIZE = 128
lr = 0.01 # using smaller learning rate is better
epoch_decay = 2e-3
weight_decay = 1e-5
margin = 1.0

# You can set use_upsampling=True and pass the class name by upsampling_cols=['Cardiomegaly'] to do upsampling. This may improve the performance
traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=True, use_frontal=True, image_size=224, mode='train', class_index=class_id)
testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=class_id)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

imratio = traindSet.imratio

# model
set_all_seeds(SEED)
model = DenseNet121(pretrained=False, last_activation=None, activations='relu', num_classes=1)
model = model.cuda()


# load pretrained model
if True:
    PATH = './ce_pretrained_model.pth' 
    state_dict = torch.load(PATH)
    state_dict.pop('classifier.weight', None)
    state_dict.pop('classifier.bias', None) 
    model.load_state_dict(state_dict, strict=False)

# define loss & optimizer
loss_fn = AUCMLoss()
optimizer = PESG(model, 
                 loss_fn=loss_fn, 
                 lr=lr, 
                 margin=margin, 
                 epoch_decay=epoch_decay, 
                 weight_decay=weight_decay)

best_val_auc = 0
for epoch in range(10):
  if epoch > 0:
     optimizer.update_regularizer(decay_factor=2)
  for idx, data in enumerate(trainloader):
      train_data, train_labels = data
      train_data, train_labels = train_data.cuda(), train_labels.cuda()
      y_pred = model(train_data)
      y_pred = torch.sigmoid(y_pred)
      loss = loss_fn(y_pred, train_labels)
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
                test_data, test_label = data
                test_data = test_data.cuda()
                y_pred = model(test_data)
                test_pred.append(y_pred.cpu().detach().numpy())
                test_true.append(test_label.numpy())
            
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            val_auc =  roc_auc_score(test_true, test_pred) 
            model.train()

            if best_val_auc < val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), 'finetuned_model.pth')
              
        print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, lr=%.4f'%(epoch, idx, val_auc,  optimizer.lr))

print ('Best Val_AUC is %.4f'%best_val_auc)