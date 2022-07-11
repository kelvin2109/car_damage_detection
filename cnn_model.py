import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np
 

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(32,32,3)
        self.fc1= nn.Linear(32*6*6,64)
        self.fc2 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(-1,32*6*6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model().to(device)

    def train(self,training_set, epochs,learning_rate = 0.05):
        
        optimizer = torch.optim.Adam(self.parameters(),lr = learning_rate)
        criterion = nn.BCELoss()

        
        #return numpy with shape epochs filled by np.nan
        losses = np.full(epochs,np.nan)
         
        n_total_steps = len(training_set)
         
        with trange(epochs) as total_epochs:
            for e in total_epochs:
                batch_loss = 0.
                for img,labels in training_set:
                    
                    img = img.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)

                    optimizer.zero_grad()
                    outputs = self.model(img)

                    loss = criterion(outputs,labels.float())
                    batch_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                
                batch_loss /= n_total_steps
                losses[e] = batch_loss
                total_epochs.set_postfix(loss="{0:.3f}".format(batch_loss))
        
        return losses

    def accuracy_score(self,input_val):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            y_pred = []
            y_true = []
            n_class_correct = [0 for i in range(2)]
            n_class_samples = [0 for i in range(2)]
            
            
            for img,labels in input_val:

                img = img.to(device) 
                labels = labels.to(device) 
                outputs = self.model(img)
                

                classes = ('damage','whole')
                predicted = torch.round(outputs).squeeze(1)
       
                y_pred += predicted.tolist()
                y_true += labels.tolist()

                n_samples += labels.size(0) 
                n_correct += (predicted == labels).sum().item()
                         

                for i in range(img.size(0)):
                    label = labels[i]
                    pred = predicted[i]

                    if(label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

                
           
            acc = 100.0 * n_correct/n_samples
            print('Accuracy of the network: {0:.3f} %'.format(acc))

            

            for i in range(2):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classes[i]}: {acc} %')

    
        
        return y_pred,y_true

    def predict(self,inputs):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
        with torch.no_grad():
                img = inputs.to(device)
                outputs = self.model(img)
                return torch.round(outputs)
               
                
