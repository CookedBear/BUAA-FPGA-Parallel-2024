
%matplotlib notebook
import matplotlib.pyplot as plt

def plot_to_notebook(time_sec, in_signal, n_samples, out_signal=None):
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.xlabel('Time (usec)')
        plt.grid()
        plt.plot(time_sec[:n_samples]*1e6,in_signal[:n_samples],'y-',label='Input signal')
        if out_signal is not None:
                plt.plot(time_sec[:n_samples]*1e6,out_signal[:n_samples],'g-',linewidth=2,label='FIR output')
        plt.legend()

import numpy as np
#模拟信号源生成输入信号
T = 0.002 # 采样时间范围
fs = 100e6 # 采样频率
n = int(T * fs) # 采样点数
t = np.linspace(0, T, n, endpoint=False) # 生成时间序列

samples = 10000*np.sin(0.2e6*2*np.pi*t) + 1500*np.cos(46e6*2*np.pi*t) + 2000*np.sin(12e6*2*np.pi*t)
# 信号源为200kHz的正弦信号，46MHz和12MHz的低幅度信号作为输入的模拟噪声。

samples = samples.astype(np.int32)   # 将样本转换为32位整数
print('Number of samples: ',len(samples))
#输出采样点数
plot_to_notebook(t,samples,1000)
#画图


# 采用SciPy函数调用的软件FIR滤波器
from scipy.signal import lfilter
coeffs = [-255,-260,-312,-288,-144,153,616,1233,1963,2739,3474,4081,4481,4620,4481,4081,3474,2739,1963,1233,616,153,-144,-288,-312,-260,-255]
# 滤波器系数向量，生成0-5MHz频带的低通滤波器。
import time
start_time = time.time()
sw_fir_output = lfilter(coeffs,70e3,samples)
stop_time = time.time()
sw_exec_time = stop_time - start_time
print('软件滤波器执行时间：',sw_exec_time) # 输出软件滤波器的执行耗时。

# 画图
plot_to_notebook(t,samples,1000,out_signal=sw_fir_output)
通过滤波，可将高于5MHz的信号过滤掉。


