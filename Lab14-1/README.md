# 背景
因为教材中的内容不全，实验无法正常进行，所以找到了原始的项目代码。见：https://github.com/CorentinJ/Real-Time-Voice-Cloning
这里就按照这个项目中的代码进行演示

# 步骤
## 1. 克隆项目
git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git

或者直接下载包


## 2. 安装环境
1. 创建环境
python create -m .Lab14_1_env
2. 进入项目
cd Real-Time-Voice-Cloning-master
注意：一定要进入项目中的存在代码的目录
3. 替换 pyproject.toml
源代码是19年的老代码了，里面的 pyproject.toml 格式太老了，无法使用 pip install . 进行自动化下载，我们需要对其进行修改。
你可以直接替换成下面的：
```txt
[project]
name = "real-time-voice-cloning"
version = "0.0.0"
requires-python = ">=3.9,<3.10"

dependencies = [
  "huggingface-hub[hf_xet]>=0.26,<1",
  "inflect==5.3.0",
  "librosa==0.9.2",
  "matplotlib==3.5.1",
  "numpy>=1.26,<2",
  "Pillow==8.4.0",
  "PyQt5==5.15.11",
  "scikit-learn==1.0.2",
  "scipy>=1.7",
  "setuptools<=80.8.0",
  "sounddevice==0.4.3",
  "soundfile==0.10.3.post1",
  "tqdm==4.62.3",
  "umap-learn==0.5.2",
  "Unidecode==1.3.2",
  "urllib3==1.26.7",
  "webrtcvad==2.0.10",
]

[project.optional-dependencies]
cpu = ["torch==1.10.*"]
cuda = ["torch==1.10.*"]

[tool.setuptools.packages.find]
where = ["."]
include = ["encoder*", "synthesizer*", "vocoder*", "toolbox*", "utils*"]
```
然后执行： pip install .
最后安装torch：
```powershell
# CPU 版（大多数人）
pip install .[cpu]
# GPU 版（有 NVIDIA 显卡）
pip install .[cuda]
```

5. 运行
python demo_cli.py --no_sound
(记得翻墙，因为要连huggingface下载模型)



# 可能遇见的问题
PS：因为我是自己用已经安装好的GPU跑的，所以会遇到一些版本问题，如下
1. AttributeError: module 'numpy' has no attribute 'cumproduct'
原因：源代码太老了，使用很旧的numpy。
解决：需要找到对应的位置进行修改
在 vocoder/models/fatchord_version.py 中，找到 total_scale = np.cumproduct(upsample_scales)[-1]
改成：total_scale = np.cumprod(upsample_scales)[-1]

2. TypeError: melspectrogram() takes 0 positional arguments but 2 positional arguments (and 2 keyword-only arguments) were given
原因：源代码太老了，使用很旧的 librosa
解决：
在 encoder/audio.py 找到
frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
修改为：（需要添加 power 这个关键参数）
frames = librosa.feature.melspectrogram(
    y=wav, 
    sr=sampling_rate,
    n_fft=int(sampling_rate * mel_window_length / 1000), 
    hop_length=int(sampling_rate * mel_window_step / 1000),
    n_mels=mel_n_channels,
    power=1
)

3. Caught exception: TypeError('resample() takes 1 positional argument but 3 were given')
原因：librosa 新版 resample 函数格式变了，老代码传参方式报错
位置：encoder/audio.py
解决：
将 wav = librosa.resample(wav, source_sr, sampling_rate) 替换为 wav = librosa.resample(y=wav, orig_sr=source_sr, target_sr=sampling_rate)