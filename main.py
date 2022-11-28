# Check out the code in work at https://www.kaggle.com/hsankesara/prototypical-net/
# Check out the blog at <COMING SOON>

import torch
import numpy as np  # linear algebra
from JointNetDrift import Joint_Prediction
from JointNetDriftVer import Joint_PredictionVer
import torch.optim as optim
from preprocessing import LoadDriftData
from config import args
from torch.utils.data import TensorDataset,DataLoader
from prototypical_batch_sampler import PrototypicalBatchSampler
import json




def init_dataset():
    # Reading the data
    print('Reading the train data')
    all_data_frame = LoadDriftData(args.Data_Vector_Length, args.DATA_FILE,args.DATA_SAMPLE_NUM)
    Drift_data_array = all_data_frame.values
    where_are_nan = np.isnan(Drift_data_array)
    where_are_inf = np.isinf(Drift_data_array)
    Drift_data_array[where_are_nan] = 0.0
    Drift_data_array[where_are_inf] = 0.0
    print(True in np.isnan(Drift_data_array))

    # random shuffle datasets
    np.random.shuffle(Drift_data_array)
    data_count = Drift_data_array.shape[0]  # data count
    train_x = Drift_data_array[0:int(data_count * args.Train_Ratio), 0:args.Data_Vector_Length]
    train_y = Drift_data_array[0:int(data_count * args.Train_Ratio), -2]
    train_locy = Drift_data_array[0:int(data_count * args.Train_Ratio), -1]
    test_x = Drift_data_array[int(data_count * args.Train_Ratio):, 0:args.Data_Vector_Length]
    test_y = Drift_data_array[int(data_count * args.Train_Ratio):, -2]
    test_locy = Drift_data_array[int(data_count * args.Train_Ratio):, -1]
    y=np.hstack((train_y,test_y))
    n_classes = len(np.unique(y))
    if n_classes < args.Nc:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return train_x,train_y,train_locy,test_x,test_y,test_locy


def init_sampler(labels):
    classes_per_it = args.Nc
    num_samples = args.Ns + args.Nq

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=args.iterations)


def init_dataloader():
    train_x,train_y,train_locy,test_x,test_y,test_locy = init_dataset()
    sampler = init_sampler(train_y)
    # TODO 生成train DataLoader
    Train_DS = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y), torch.unsqueeze(torch.FloatTensor(train_locy),1))
    train_dataloader = DataLoader(Train_DS, batch_sampler=sampler)
    # TODO 生成test DataLoader
    sampler = init_sampler(test_y)
    Test_DS = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y), torch.unsqueeze(torch.FloatTensor(test_locy),1))
    test_dataloader = DataLoader(Test_DS, batch_sampler=sampler)
    return train_dataloader,test_dataloader

def main():

    ModelSelect = args.FAN # 'RNN', 'FAN' , 'FNN','FQN','FAN'
    train_dataloader,test_dataloader=init_dataloader()

    print('Checking if GPU is available')
    use_gpu = torch.cuda.is_available()
    # use_gpu = False

    # Set training iterations and display period
    num_episode = args.num_episode

    # Initializing prototypical net
    print('Initializing Joint_Prediction net')
    model = Joint_Prediction(use_gpu=use_gpu, Data_Vector_Length =args.Data_Vector_Length, ModelSelect=ModelSelect)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = init_lr_scheduler(optimizer)

    train_loss = []
    train_class_acc = []
    train_loc_acc = []
    test_loss = []
    test_class_acc = []
    test_loc_acc = []
    centroid_matrix=torch.Tensor()
    # Training loop
    for i in range(num_episode):
        model.train()
        for batch_idx,data in enumerate(train_dataloader):
            optimizer.zero_grad()
            datax,datay,locy = data
            # locy = torch.unsqueeze(locy, 1)
            loss,class_acc,loc_acc,centroid_matrix = model(datax,datay,locy)
            # loss = loc_loss_fun(pre_loc_y,locy)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_class_acc.append(class_acc.item())
            train_loc_acc.append(loc_acc.item())
            centroid_matrix=centroid_matrix

        avg_loss = np.mean(train_loss)
        avg_class_acc = np.mean(train_class_acc)
        avg_loc_acc = np.mean(train_loc_acc)
        print('{} episode,Avg Train Loss: {}, Avg Train Class Acc: {}, Avg Train loc Acc: {}'.format(
            i,avg_loss, avg_class_acc,avg_loc_acc))

        lr_scheduler.step()

    PATH = './input/Model/JPN/'+args.DATA_FILE+'/{name}_model_embeding.pkl'.format(name=ModelSelect)
    torch.save(model.state_dict(), PATH)
    # save centroid matrix
    torch.save(centroid_matrix, './input/Model/JPN/' + args.DATA_FILE + '/{name}_centroid_matrix.pt'.format(name=ModelSelect))
    # save loss
    with open('./input/Model/JPN/' + args.DATA_FILE + '/{name}_loss.json'.format(name=ModelSelect), "w") as f:
        json.dump({"train_loss":np.array(train_loss[len(train_loss)-5:]).sum()/len(train_loss[len(train_loss)-5:]),
                   "train_class_acc":np.array(train_class_acc[len(train_class_acc)-5:]).sum()/len(train_class_acc[len(train_class_acc)-5:]),
                   "train_loc_acc":np.array(train_loc_acc[len(train_loc_acc)-5:]).sum()/len(train_loc_acc[len(train_loc_acc)-5:])}, f)
    # Test loop
    for batch_idx, data in enumerate(test_dataloader):
        datax,datay,locy = data
        loss,class_acc,loc_acc,centroid_matrix =model(datax,datay,locy)
        test_loss.append(loss.item())
        test_class_acc.append(class_acc.item())
        test_loc_acc.append(loc_acc.item())
    avg_class_acc = np.mean(test_class_acc)
    avg_loc_acc = np.mean(test_loc_acc)
    # save result
    with open('./input/Model/JPN/' + args.DATA_FILE + '/{name}_result.json'.format(name=ModelSelect), "w") as f:
        json.dump({"avg_class_acc":avg_class_acc,"avg_loc_acc":avg_loc_acc}, f)

    print(' Avg Test Class Acc: {}, Avg Test loc Acc: {}'.format(
        avg_class_acc, avg_loc_acc))


def init_lr_scheduler(optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=args.lr_scheduler_gamma,
                                           step_size=args.lr_scheduler_step)


def model_verification():
    ModelSelect = args.FAN  # 'RNN', 'FAN', 'Seq2Seq'
    train_dataloader, test_dataloader = init_dataloader()

    print('Checking if GPU is available')
    use_gpu = torch.cuda.is_available()
    BASE_PATH = './input/Model/JPN/' + args.DATA_FILE
    model = Joint_PredictionVer(use_gpu=use_gpu, Data_Vector_Length=args.Data_Vector_Length, ModelSelect=ModelSelect)
    PATH = BASE_PATH + "/" + ModelSelect + '_model_embeding.pkl'
    model.load_state_dict(torch.load(PATH))
    # Test loop
    test_class_acc = []
    test_loc_acc = []
    for batch_idx, data in enumerate(test_dataloader):
        datax, datay, locy = data
        type_acc, loc_acc = model(datax,datay,locy, BASE_PATH)
        test_class_acc.append(type_acc)
        test_loc_acc.append(loc_acc.item())
    avg_class_acc = np.mean(test_class_acc)
    avg_loc_acc = np.mean(test_loc_acc)

    print(' Avg Test Class Acc: {}, Avg Test loc Acc: {}'.format(
        avg_class_acc, avg_loc_acc))



if __name__ == "__main__":

    main()
