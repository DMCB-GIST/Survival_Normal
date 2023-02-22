import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
import argparse
pd.options.mode.chained_assignment = None
import random

random_seed = 0
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)

parser = argparse.ArgumentParser()

parser.add_argument('--input', '-i', type=str, default=False)
parser.add_argument('--surv_input', '-s', type=str, default="surv_df.tsv")
parser.add_argument('--output', '-o', type=str, default=False)
parser.add_argument('--cancer', '-c', type = str)
parser.add_argument('--feature', '-d', type = int, default = 0) # How many features do you want to use for selection process? (Input data must be sorted by importance)


config = parser.parse_args()
cancer = config.cancer
feature_num = config.feature
    
cuda = True if torch.cuda.is_available() else False
if cuda:
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

## Loss Function ##
class PartialNLL(torch.nn.Module):
    def __init__(self):
        super(PartialNLL, self).__init__()

    def _make_R_rev(self, y):
        R = np.zeros((y.shape[0], y.shape[0]))
        y = y.detach().cpu().numpy()
        for i in range(y.shape[0]):
            idx = np.where(y >= y[i])
            R[i, idx] = 1
        return torch.tensor(R)

    def forward(self, theta, time, observed):
        R = self._make_R_rev(time).to(theta.device)
        exp_theta = torch.exp(theta)
        num_observed = torch.sum(observed)
        loss = -torch.sum((theta.reshape(-1) - torch.log(torch.sum((exp_theta * R.t()), 0))) * observed) / num_observed
        if np.isnan(loss.data.tolist()):
            for a, b in zip(theta, exp_theta):
                print(a, b)
        return loss

## Neural Network model ##
class NN(nn.Module):
    def __init__(self, nfeat, n_hidden):
        super(NN, self).__init__()
        self.hidden_dim = n_hidden
        self.fc1 = torch.nn.Linear(nfeat, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = torch.nn.Linear(self.hidden_dim, 1)

        self.ReLU = torch.nn.ReLU()
        self.Tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.Tanh(self.fc2(x))
        x = self.fc3(x)
        return x



def main(): 	
    ## Data Load ## 

    surv_df = pd.read_table("./../data/"+cancer+'/'+config.surv_input)
    surv_time = torch.tensor(surv_df['time'].to_numpy(),dtype=torch.float, requires_grad=True, device=device)
    surv_status = torch.tensor(surv_df['status'].to_numpy(), device=device)

    X = pd.read_table("./../data/"+cancer+'/'+config.input, delimiter="\t", header=None)
    stdscaler = StandardScaler()
    X = stdscaler.fit_transform(np.array(X))
    X = torch.tensor(X, device=device, dtype=torch.float)    

    feature_num = config.feature
    feature_num = X.shape[1] if feature_num == 0 else feature_num
    X = X[:,:feature_num]    

    ## Training Config ##
    print_interval = 25
    n_epoch = 100
    lr = 0.0005
    wd = 1e-5

    ## Train ## 
    print(X.shape)
    perf_list = []
    for iteration in range(20):
        
        feature_normal = {}
        feature_tumor = {}
        print("Iteration", iteration + 1)

        ## Data Split ## 
        train_idx, test_idx, _, _ = train_test_split(np.arange(X.shape[0]), surv_status, test_size=.3, random_state=iteration)

        X_train = X[train_idx,:]
        X_test = X[test_idx,:]

        surv_time_train = surv_time[train_idx]
        surv_time_test = surv_time[test_idx]

        surv_status_train = surv_status[train_idx]
        surv_status_test = surv_status[test_idx]

        hidden_dim = int(np.sqrt(X_train.shape[1]))

        model = NN(X_train.shape[1], hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = PartialNLL()
        
        for epoch in range(n_epoch):
            model.train()
            output = model.forward(X_train)
            loss = criterion(output,surv_time_train, surv_status_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((epoch + 1) % print_interval == 0):
                model.eval()
                print(epoch + 1, loss.item(),

                      "Survival Train", np.round(concordance_index(event_times=surv_time_train.cpu().detach().numpy(),
                                                                   event_observed=surv_status_train.cpu().detach().numpy(),
                                                                   predicted_scores=-model.forward(X_train).cpu().detach().numpy()), 3),
                      "Survival Test", np.round(concordance_index(event_times=surv_time_test.cpu().detach().numpy(),
                                                                  event_observed=surv_status_test.cpu().detach().numpy(),
                                                                  predicted_scores=-model.forward(X_test).cpu().detach().numpy()), 3))


        test_perf = concordance_index(event_times=surv_time_test.cpu().detach().numpy(),
                                        event_observed=surv_status_test.cpu().detach().numpy(),
                                        predicted_scores=-model.forward(X_test).cpu().detach().numpy())
        print("Test Performance : ", test_perf)
        perf_list.append(test_perf)

        print("Avg.", np.mean(perf_list),"±",np.std(perf_list))
        if config.output:
            torch.save(perf_list, config.output)    

if __name__ == '__main__':
    main()
    
