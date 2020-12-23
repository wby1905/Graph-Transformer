import random
from collections import Counter

import nni
import numpy as np
import torch as t
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import DataListLoader
from torch_geometric.nn import GCNConv, DataParallel
from torchnet import meter
from tqdm import tqdm

from config import opt
from data.load_data import TUD
from model.EarlyStopping import EarlyStopping
from model.NewGNN import GNN
from model.WarmUp import GradualWarmupScheduler



def train():
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        t.manual_seed(opt.seed)
        t.cuda.manual_seed(opt.seed)
        cudnn.deterministic = False
        cudnn.benchmark = True

    # data
    global train_loader, val_loader, criterion, sample_weight, dataset, val_idx, train_idx, indices, split, val_accuracy, accuracy
    dataset = TUD()
    indices = list(range(len(dataset)))
    if opt.dataset == 'PROTEINS':
        class_idx = {c: [graph_id for graph_id in range(len(dataset)) if dataset[graph_id].y.item() == c] for c in
                     range(opt.classes)}
        aug = np.random.choice(class_idx[1], len(class_idx[0]) - len(class_idx[1])).tolist()
        for i in aug:
            indices.append(i)
    if opt.dataset == 'PTC':
        class_idx = {c: [graph_id for graph_id in range(len(dataset)) if dataset[graph_id].y.item() == c] for c in
                     range(opt.classes)}
        aug = np.random.choice(class_idx[1], len(class_idx[0]) - len(class_idx[1])).tolist()
        for i in aug:
            indices.append(i)
    if opt.dataset == 'MUTAG':
        indices = list(range(len(dataset)))
        class_idx = {c: [graph_id for graph_id in range(len(dataset)) if dataset[graph_id].y.item() == c] for c in
                     range(opt.classes)}
        aug = np.random.choice(class_idx[0], len(class_idx[1]) - len(class_idx[0])).tolist()
        for i in aug:
            indices.append(i)
    split = int(np.floor(opt.val_split * len(dataset)))
    # train_idx, val_idx = indices[split:], indices[:split]
    np.random.shuffle(indices)
    # k折验证
    train_loss_sum, val_loss_sum = 0, 0
    train_acc_sum, val_acc_sum = 0, 0
    for fold in range(opt.k_fold):
        # t.autograd.set_detect_anomaly(True)
        # 配置 model
        model = GNN(opt)
        if fold == 0:
            print(model)
            print('# model parameters:', sum(param.numel() for param in model.parameters()))

        # 初始化
        for m in model.modules():
            if isinstance(m, GCNConv):
                t.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, t.nn.Linear) or isinstance(m, t.nn.Conv1d):
                t.nn.init.kaiming_uniform_(m.weight)
        model.to(opt.device)
        device_ids = range(t.cuda.device_count())

        if len(device_ids) > 1:
            model = DataParallel(model)

        # 创建训练集以及验证集
        train_idx, val_idx = [], []
        fold_size = len(indices) // opt.k_fold
        for j in range(opt.k_fold):
            idx = slice(fold_size * j, fold_size * (j + 1))
            if j == fold:
                val_idx = indices[idx]
            else:
                train_idx += indices[idx]
        train_idx += indices[fold_size * opt.k_fold:]
        counter = Counter([dataset[i].y.item() for i in train_idx])
        counter = sorted(counter.items(), key=lambda x: x[0])
        # sample_weight = t.nn.functional.softmax(t.tensor([len(dataset) / i[1] for i in counter]), dim=0).detach()
        print(counter)
        # print(train_idx)

        # 开始训练
        # 多进程需要写name==main
        if __name__ == '__main__':
            
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataListLoader(dataset, batch_size=opt.batch_size * len(device_ids), pin_memory=True,
                                      num_workers=opt.num_workers,
                                      sampler=train_sampler)
            val_loader = DataListLoader(dataset, batch_size=opt.batch_size * len(device_ids), pin_memory=True,
                                    sampler=val_sampler)

            # 损失函数以及优化器
            lr_warmup = opt.lr * 0.1
            optimizer = t.optim.Adam(model.parameters(), lr=lr_warmup, weight_decay=opt.weight_decay)
            scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', opt.lr_decay, min_lr=1e-7)
            scheduler_cosine = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, 1e-7)
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10,
                                                      after_scheduler=scheduler)
            # 指标
            val_loss_meter = meter.AverageValueMeter()
            loss_meter = meter.AverageValueMeter()
            confusion_matrix = meter.ConfusionMeter(opt.classes)
            early_stopping = EarlyStopping(patience=30,delta=0)
            for epoch in range(opt.max_epoch):
                loss_meter.reset()
                confusion_matrix.reset()

                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    output, loss, labels = model(data) 
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    # 更新指标
                    loss_meter.add(loss.item())
                    confusion_matrix.add(output.detach(), labels)

                    if (i + 1) % opt.print_freq == 0:
                        print('loss now:', loss_meter.value()[0])

                # t.save(model.state_dict(), f='./checkpoints/' + time.strftime('%m%d_%H:%M.pth'))

                val_cm, val_accuracy, val_loss = val(model, val_loader)
                val_loss_meter.add(val_loss.item())
                confusion_value = confusion_matrix.value()
                accuracy = 0
                for i in range(opt.classes):
                    accuracy += 100. * confusion_value[i][i] / confusion_value.sum()

                print('\ntrain_accuracy', accuracy)
                print('\nval_accuracy', val_accuracy)
                # if epoch > 20:
                #     scheduler.step(val_accuracy)
                # else:
                #     lr_warmup = opt.lr * 0.1 * (epoch//2+1)
                #     optimizer.param_groups[0]['lr'] = lr_warmup
                lr = optimizer.param_groups[0]['lr']
                scheduler_warmup.step(epoch, val_accuracy)
                # nni.report_intermediate_result(val_accuracy)
                if epoch > opt.max_epoch / 4:
                    early_stopping(-val_accuracy, model)
                print('fold:{}, epoch:{}, loss:{}, lr:{}, '
                      '\n train_cm:\n{}, \n val_loss:{}, val_cm:\n{}'
                      .format(fold,
                              epoch,
                              loss_meter.value()[0],
                              lr,
                              str(confusion_matrix.value()),
                              val_loss_meter.value()[0],
                              str(val_cm.value())))
                if early_stopping.early_stop or (epoch > 50 and early_stopping.best_score < 65):
                    print("Early stopped")
                    break
               

            val_acc_sum += early_stopping.best_score
            train_acc_sum += accuracy
            
        nni.report_intermediate_result(val_acc_sum / (fold+1))
        if val_acc_sum / (fold+1) < 65:
            nni.report_final_result(val_acc_sum / (fold + 1))
            return
    print('k-fold train accuracy: ', train_acc_sum / opt.k_fold, '\n', 'val accuracy: ', val_acc_sum / opt.k_fold)
#     nni.report_final_result(val_acc_sum / opt.k_fold)
    return train_acc_sum, val_acc_sum


@t.no_grad()
def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(opt.classes)
    for i, val_x in enumerate(dataloader):
        output, loss, val_y = model(val_x) 
        loss = loss.mean()
        confusion_matrix.add(output.detach(), val_y.detach())

    model.train()
    confusion_value = confusion_matrix.value()
    accuracy = 0
    for i in range(opt.classes):
        accuracy += 100. * confusion_value[i][i] / confusion_value.sum()
    return confusion_matrix, accuracy, loss


if __name__ == '__main__':
    t.multiprocessing.set_sharing_strategy('file_system')
    
#     RECEIVED_PARAMS = nni.get_next_parameter()
    
#     opt.top_k = RECEIVED_PARAMS['top_k']
#     opt.K = RECEIVED_PARAMS['K']
#     opt.m = RECEIVED_PARAMS['m']
#     opt.conv_num = opt.K - 1
#     opt.gcn_num = RECEIVED_PARAMS['gcn_num']
    
#     opt.conv_channel = RECEIVED_PARAMS['conv_channel']
#     opt.conv1d_channel = RECEIVED_PARAMS['conv1d_channel']
#     opt.fc_channel = RECEIVED_PARAMS['fc_channel']
#     opt.fc_layers_num = RECEIVED_PARAMS['fc_layers_num']
    
#     opt.lr = RECEIVED_PARAMS['lr']
#     opt.weight_decay = RECEIVED_PARAMS['weight_decay']
#     opt.drop_out = RECEIVED_PARAMS['drop_out']
#     opt.eps = RECEIVED_PARAMS['eps']

#     opt.batch_size = RECEIVED_PARAMS['batch_size']
    train()
    matplotlib

#     t_acc = []
#     v_acc = []
#     for i in range(10):
#         t_a, v_a = train()
#         opt.seed += 6
#         t_acc.append(t_a)
#         v_acc.append(v_a)
#     print(t_acc)
#     print(v_acc)
#     print(sum(t_acc) / 100, sum(v_acc) / 100)
    
    
    
#     a = TUD()
#     for i in range(len(a)):
#         print(a[i].x_w_trees.shape)
#     for i in range(2,5):
#         for j in range(2,10):
#             opt.K = i
#             opt.m = j
#             a = TUD()
#             print(i,j)
    