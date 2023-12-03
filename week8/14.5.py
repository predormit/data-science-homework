import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 读取音频文件
filename = "ITZY (있지) - None of My Business (MV版) (Single Version).wav"  # 替换成你的音频文件路径
sample_rate, data = wavfile.read(filename)

# 取一个音频片段（可根据需要进行调整）
start = 0  # 起始时间（单位：秒）
duration = 5  # 音频时长（单位：秒）
end = start + duration

# 提取选择的音频片段
segment = data[int(start * sample_rate): int(end * sample_rate)]

# 进行快速傅里叶变换
fft = np.fft.fft(segment)

# 计算频率轴
freq_axis = np.fft.fftfreq(len(segment), 1/sample_rate)

# 可视化傅里叶变换的结果
plt.figure(figsize=(10, 4))
plt.plot(freq_axis, np.abs(fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT Result')
plt.grid(True)
plt.show()