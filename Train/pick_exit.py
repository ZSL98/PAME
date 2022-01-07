import numpy as np
import json
import matplotlib.pyplot as plt

dataset = 'imagenet'
batch_size = 32
metric_thres = 99
latency_satisfied_batch_cnt = [0]*33
latency_satisfied_batch_ratio = [0]*33

# moveon_ratio_thres = [0.8125,0.75,0.75]+[0.8125]*20+[0.75]+[0.6875]*2+[0.625]+[0.5625]*2+[0.4375]*2+[0.875]*2

moveon_ratio_thres = [0.8125,0.75,0.75]+[0.8125]*20+[0.75]+[0.6875]*2+[0.625]+[0.5625]*2+[0.4375]*2+[0.875]*2

for exit in range(1,33,3):
    ori_hist_data = []
    if dataset in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']:
        with open('./moveon_dict/bert/{}/{}_exit_l[0, 5, {}]_b{}_t{}.json'.format(dataset, dataset, exit, batch_size, metric_thres), 'rb') as f:
            last_moveon_dict = json.load(f)
    elif dataset in ['imagenet', 'imagenette', 'imagewoof']:
        with open('./moveon_dict/resnet/{}/resnet_exit_l[0, {}]_b{}_t{}.json'.format(dataset, exit, batch_size, metric_thres), 'rb') as f:
            last_moveon_dict = json.load(f)
    else:
        with open('./moveon_dict/{}/{}_trainall_l[0, {}]_b{}_t{}.json'.format(dataset, dataset, exit, batch_size, metric_thres), 'rb') as f:
            last_moveon_dict = json.load(f)       

    if dataset in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']:
        last_moveon_dict = [last_moveon_dict['0'][i:i+batch_size] for i in range(0,len(last_moveon_dict['0']), batch_size)]
        for i in range(len(last_moveon_dict)):
            ori_hist_data.append(sum(last_moveon_dict[i]))
    else:
        i = 0
        for k,v in last_moveon_dict.items():
            ori_hist_data.append(sum(last_moveon_dict[k]))
            i = i+1

    hist_data = plt.hist(ori_hist_data, bins=16, range=(0,batch_size))
    print(exit, '\t', [int(x) for x in hist_data[0]])
    # hist_data = np.load('./moveon_dict/posenet_hist_data_e{}_b{}.npy'.format(exit, batch_size))
    hist_data = hist_data[0]
    for i in range(len(hist_data)):
        if hist_data[i] < moveon_ratio_thres[exit]:
            latency_satisfied_batch_cnt[exit] = latency_satisfied_batch_cnt[exit] + 1
    latency_satisfied_batch_cnt[exit] = latency_satisfied_batch_cnt[exit]/len(hist_data)

print(latency_satisfied_batch_cnt)