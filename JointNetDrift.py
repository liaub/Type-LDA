import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import args


class Joint_Prediction(nn.Module):
    '''
    Joint prediction model
    1.Predict possible drift categories(PrototypicalNet)
    2.Predict at which point the drift is likely to occur(DriftPointNet)
    '''
    def __init__(self,use_gpu,Data_Vector_Length,ModelSelect):
        super(Joint_Prediction,self).__init__()
        self.use_gpu=use_gpu
        self.Ns = args.Ns
        self.Nc = args.Nc
        self.Nq = args.Nq
        self.Data_Vector_Length=Data_Vector_Length
        self.ModelSelect=ModelSelect
        self.loc_loss_fun =nn.MSELoss()
        self.PrototypicalNet = PrototypicalNet(use_gpu=self.use_gpu, Data_Vector_Length=self.Data_Vector_Length, ModelSelect=self.ModelSelect)
        self.DriftPointNet = DriftPointNet(use_gpu=self.use_gpu,Data_Vector_Length=self.Data_Vector_Length)
        # self.OnlyDriftPointNet = OnlyDriftPointNet(use_gpu=self.use_gpu, Data_Vector_Length=self.Data_Vector_Length)
        self.AutomaticWeightedLoss= AutomaticWeightedLoss(2)


    def forward(self,datax,datay,locy):
        #Get input data (center of mass)
        input = self.PrototypicalNet(datax)
        class_loss, class_acc ,centroid_matrix=self.prototypical_loss(input,datay,self.Ns)
        pre_loc_y = self.DriftPointNet(input, datax)
        # pre_loc_y=self.OnlyDriftPointNet(datax)
        loc_loss=self.loc_loss_fun(pre_loc_y,locy)
        loc_acc=self.cal_loc_acc(pre_loc_y,locy)
        # Combination of two Loss
        loss = self.AutomaticWeightedLoss(class_loss, loc_loss)
        return loss,class_acc,loc_acc,centroid_matrix

    def cal_loc_acc(self,pre_loc_y,locy):
        '''
        R**2 :cal loc acc
        where u is the residual sum of squares
         ((y_true - y_pred) ** 2).sum() and v is the total
            sum of squares ((y_true - y_true.mean()) ** 2).sum().
        '''
        u = torch.sum(torch.abs(locy - pre_loc_y) ** 2)
        v = torch.sum(torch.abs(locy - torch.mean(locy)) ** 2)
        acc_loc = 1 - u / v
        return acc_loc


    def prototypical_loss(self,input, target, n_support):
        '''
        Compute the barycentres by averaging the features of n_support
        samples for each class in target, computes then the distances from each
        samples' features to each one of the barycentres, computes the
        log_probability for each n_query samples for each one of the current
        classes, of appartaining to a class c, loss and accuracy are then computed
        and returned
        Args:
        - input: the model output for a batch of samples
        - target: ground truth for the above batch of samples
        - n_support: number of samples to keep in account when computing
          barycentres, for each one of the current classes
        '''
        target_cpu = target.to('cpu')
        input_cpu = input.to('cpu')

        def supp_idxs(c):
            # FIXME when torch will support where as np
            return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

        # FIXME when torch.unique will be available on cuda too
        classes = torch.unique(target_cpu)
        n_classes = len(classes)
        # FIXME when torch will support where as np
        # assuming n_query, n_target constants
        A = classes[0].item()
        n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

        support_idxs = list(map(supp_idxs, classes))

        prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

        # FIXME when torch will support where as np
        query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

        query_samples = input.to('cpu')[query_idxs]
        dists = euclidean_dist(query_samples, prototypes)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

        return loss_val, acc_val,prototypes

    def assign_network_arameters(self,variateList):
        datax = variateList[0]
        datay = variateList[1]
        datay, locy_ = datay.split(1, dim=1)
        locy_ = torch.squeeze(locy_)
        loc_numpy = locy_.numpy()
        locy = []
        for value in loc_numpy:
            if float(value) > 0:
                locy.append((float(value) - 1) / (self.Data_Vector_Length - 1))
            else:
                locy.append((3 - 1) / (self.Data_Vector_Length - 1))
        locy = torch.from_numpy(np.array(locy))
        locy = torch.unsqueeze(locy, 1)
        total_classes = np.unique(datay)
        return datax, datay,locy,total_classes



class PrototypicalNet(nn.Module):
    def __init__(self, use_gpu=True, Data_Vector_Length=100, ModelSelect='FAN'):
        super(PrototypicalNet, self).__init__()
        self.gpu  = use_gpu
        if ModelSelect=='FNN':
            self.f = FNN(Data_Vector_Length)
        elif ModelSelect=='FAN':
            self.f = FAN(Data_Vector_Length)
        elif ModelSelect == 'FQN':
            self.f = FQN(Data_Vector_Length)
        elif ModelSelect=='FAN':
            self.f = FCN(Data_Vector_Length)
        elif ModelSelect == 'RNN':
            self.f = LSTM(Data_Vector_Length)
        elif ModelSelect == 'PNN':
            self.f = FNN(Data_Vector_Length)
        else:
            self.f = CNN(Data_Vector_Length)
        if self.gpu:
            self.f = self.f.cuda()

    def forward(self, datax):
        input = self.f(datax)
        return input



class FNN(nn.Module):
    """
    normal feed-network
    """

    def __init__(self,Data_Vector_Length):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(Data_Vector_Length, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)
        self.fc4 = nn.Linear(250, args.centord_Vector_Length)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class FAN(nn.Module):
    """
    attention feed-attention-network
    """

    def __init__(self,Data_Vector_Length):
        super(FAN, self).__init__()
        if Data_Vector_Length > 50 :
            self.hidden_size = Data_Vector_Length * 2
        else:
            self.hidden_size = Data_Vector_Length * 3
        self.attn = nn.Linear(Data_Vector_Length, self.hidden_size)
        self.W_s = nn.Linear(Data_Vector_Length, self.hidden_size)
        # self.dropout = nn.Dropout()
        self.attn_combine = nn.Linear(self.hidden_size * 2, args.centord_Vector_Length)


    def forward(self, x):
        attn_weights = F.softmax(self.attn(x))#(40,200)
        q_s= F.relu(self.W_s(x))
        attn_applied = torch.mul(attn_weights,q_s)#(40,200)
        # Attention output
        combine = torch.cat((q_s, attn_applied), 1)#(40,400)
        # combine_d = self.dropout(combine)
        output = self.attn_combine(combine)
        return output

class FQN(nn.Module):
    """
    encoded_mask ,feed-query-network
    """

    def __init__(self,Data_Vector_Length):
        super(FQN, self).__init__()
        self.Enbedding = nn.Linear(Data_Vector_Length, Data_Vector_Length)
        self.W_s = nn.Linear(Data_Vector_Length, Data_Vector_Length * 2)
        self.out = nn.Linear(Data_Vector_Length * 2, args.centord_Vector_Length)

    def forward(self, x):
        enbed_x = F.relu(self.Enbedding(x))#(40,100)
        q_s = F.tanh(self.W_s(enbed_x))#(40,200)
        encoded_mask = torch.ones_like(q_s) * -100
        attn_combine = q_s + encoded_mask
        out = self.out(attn_combine)

        return out

class LSTM(nn.Module):
    def __init__(self,Data_Vector_Length):
        super(LSTM, self).__init__()
        self.Data_Vector_Length=Data_Vector_Length
        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=1,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
            batch_first=True,  # (batch, time_step, input_size)
        )
        self.out = nn.Linear(32*2,args.centord_Vector_Length)
    def forward(self, x):
        m_t=x.reshape(-1,self.Data_Vector_Length,1)
        r_out, (h_n, h_c) = self.rnn(m_t, None)
        out = self.out(r_out[:, -1, :])
        return out

class CNN(nn.Module):
    def __init__(self,Data_Vector_Length):
        super(CNN, self).__init__()
        self.Data_Vector_Length=Data_Vector_Length
        self.Dim = 10#input_len
        self.Cla = 100 #output_size
        self.Ci = 1  #input_channel_size
        self.Knum = 16#output_channel_size
        self.Ks = [1,2,3]  ## kernel list，like [6,8,10]
        self.convs = nn.ModuleList([nn.Conv2d(self.Ci, self.Knum, (K, self.Dim)) for K in self.Ks])  ## 卷积层
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(len(self.Ks) * self.Knum, args.centord_Vector_Length)  #full_linear
        # self.lstm=LSTM(Knum*len(Ks))

    def forward(self, x):
        x = x.view(-1, self.Dim, int(self.Data_Vector_Length/self.Dim))
        x = x.unsqueeze(1)  # (40,1,100,100)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        x = torch.cat(x, 1)  # (N,Knum*len(Ks))
        x = self.dropout(x)
        y = self.fc(x)
        # return F.log_softmax(y,dim=-1)

        return y


class FCN(nn.Module):
    def __init__(self,Data_Vector_Length):
        super(FCN, self).__init__()
        self.Dim=Data_Vector_Length
        self.conv_kernal_size=5
        self.pool_kernal_size=2
        self.stride=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=self.conv_kernal_size,stride=self.stride)#(16,1,96)
        self.max_pool1 = nn.MaxPool1d(kernel_size=self.pool_kernal_size)#(16,48)
        self.conv2 = nn.Conv1d(16, 32, self.conv_kernal_size, self.stride)#(32,44)
        self.max_pool2 = nn.MaxPool1d(kernel_size=self.pool_kernal_size)#(32,22)
        self.conv3 = nn.Conv1d(32, 64, self.conv_kernal_size, self.stride)#(64,18)
        self.max_pool3 = nn.MaxPool1d(kernel_size=self.pool_kernal_size)  # (64,9)
        self.cvo1=((self.Dim-self.conv_kernal_size)/self.stride)+1
        self.mpl1 = self.cvo1/self.pool_kernal_size
        self.cvo2 = ((self.mpl1 - self.conv_kernal_size) / self.stride) + 1
        self.mpl2 = self.cvo2 / self.pool_kernal_size
        self.cvo3 = ((self.mpl2 - self.conv_kernal_size) / self.stride) + 1
        self.mpl3 = int(self.cvo3 / self.pool_kernal_size)
        self.dropout = nn.Dropout()
        self.liner1 = nn.Linear(64 * self.mpl3,args.centord_Vector_Length)
        # self.liner2 = nn.Linear(200, self.Dim)
        # self.liner3 = nn.Linear(200, self.Dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool3(x)
        x = x.view(-1, 64 * self.mpl3)
        x = self.dropout(x)
        x = self.liner1(x)

        # x = F.relu(self.liner2(x))
        # x = self.liner2(x)
        # x =self.liner3(x)

        return x

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class DriftPointNet1(nn.Module):
    def __init__(self, use_gpu=True,Data_Vector_Length=100):
        super(DriftPointNet1, self).__init__()
        self.Data_Vector_Length = Data_Vector_Length
        self.centord_Vector_Length = args.centord_Vector_Length
        self.fc1 = nn.Linear(self.Data_Vector_Length + self.centord_Vector_Length, 800)
        # self.fc2 = nn.Linear(Data_Vector_Length, 800)

        self.fc3 = nn.Linear(800, 800)
        # self.dropout = nn.Dropout()
        self.out = nn.Linear(800, 1)


    def forward(self,centroid,datax):
        x = torch.cat((datax, centroid), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        pre_loc_y = self.out(x)

        return pre_loc_y

class DriftPointNet(nn.Module):
    def __init__(self, use_gpu=True,Data_Vector_Length=100):
        super(DriftPointNet, self).__init__()
        self.Data_Vector_Length = Data_Vector_Length
        self.centord_Vector_Length = args.centord_Vector_Length
        self.Embedding = nn.Linear(self.Data_Vector_Length, 800)
        self.mlp1 = nn.Linear(self.Data_Vector_Length + self.centord_Vector_Length, 800)
        self.mlp2 = nn.Linear(800, 800)
        # self.dropout = nn.Dropout()
        self.out = nn.Linear(800, 1)
    def forward(self,centroid,datax):
        v_x = F.relu(self.Embedding(datax))
        j_x = torch.cat((datax, centroid), dim=1)
        j2_x = F.relu(self.mlp1(j_x))
        r_x = F.relu(self.mlp2(j2_x))
        z_x = v_x + r_x
        pre_loc_y = self.out(z_x)
        return pre_loc_y

class OnlyDriftPointNet(nn.Module):
    def __init__(self,use_gpu=True,Data_Vector_Length=100):
        super(OnlyDriftPointNet, self).__init__()
        self.Data_Vector_Length = Data_Vector_Length
        self.centord_Vector_Length = args.centord_Vector_Length
        self.Embedding = nn.Linear(self.Data_Vector_Length, 800)
        self.mlp2 = nn.Linear(800, 800)
        # self.dropout = nn.Dropout()
        self.out = nn.Linear(800, 1)
    def forward(self,datax):
        v_x = F.relu(self.Embedding(datax))
        j2_x = F.relu(self.mlp2(v_x))
        pre_loc_y = self.out(j2_x)

        return pre_loc_y


def load_weights(filename, protonet, use_gpu):
    if use_gpu:
        protonet.load_state_dict(torch.load(filename))
    else:
        protonet.load_state_dict(torch.load(filename), map_location='cpu')
    return protonet


