#硬件滤波器实现
from pynq import Overlay, Xlnk
import pynq.lib.dma
import numpy as np
import time

overlay = Overlay('./accel.bit')  # 加载overlay文件
dma = overlay.filter.fir_dma  # 加载滤波器的DMA

# 为输入和输出信号分配缓冲区
xlnk = Xlnk()
# 申请内存地址连续的numpy数组作为输入输出缓冲区
in_buffer = xlnk.cma_array(shape=(n,), dtype=np.int32)
out_buffer = xlnk.cma_array(shape=(n,), dtype=np.int32)

# 将信号拷贝至输入缓冲区
np.copyto(in_buffer,samples)

start_time = time.time() # 读取开始的时间点
dma.sendchannel.transfer(in_buffer)
dma.recvchannel.transfer(out_buffer)
dma.sendchannel.wait()
dma.recvchannel.wait()
stop_time = time.time()  # 读取结束的时间点
hw_exec_time = stop_time-start_time
print('硬件滤波器执行时间: ',hw_exec_time)  #输出硬件滤波器的滤波耗时
print('硬件加速效果: ',sw_exec_time / hw_exec_time) #输出加速比

# 画图
plot_to_notebook(t,samples,1000,out_signal=out_buffer)

# 释放缓冲区
in_buffer.close()
out_buffer.close()
