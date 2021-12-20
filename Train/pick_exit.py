import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
latency_satisfied_batch_cnt = [0]*33
latency_satisfied_batch_ratio = [0]*33

moveon_ratio_thres = [0.8125,0.75,0.75]+[0.8125]*20+[0.75]+[0.6875]*2+[0.625]+[0.5625]*2+[0.4375]*2+[0.875]*2

for exit in range(1,33):
    hist_data = np.load('./moveon_dict/resnet_hist_data_e{}_b{}.npy'.format(exit, batch_size))
    hist_data = hist_data.tolist()
    for i in range(len(hist_data)):
        if hist_data[i] < moveon_ratio_thres[exit]:
            latency_satisfied_batch_cnt[exit] = latency_satisfied_batch_cnt[exit] + 1
    latency_satisfied_batch_cnt[exit] = latency_satisfied_batch_cnt[exit]/len(hist_data)

print(latency_satisfied_batch_cnt)