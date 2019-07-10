FROM ufoym/deepo:all-jupyter-py36-cu90
MAINTAINER yuepaang@gmail.com

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
RUN conda config --set show_channel_urls yes

RUN conda install jupyter -y --quiet
RUN conda install tqdm -y --quiet
RUN conda install pytorch=0.4.1 torchvision
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com tensorflow tensorboardX
RUN pip install -i http://pypi.douban.com/simple/ cupy pynvrtc --trusted-host pypi.douban.com
RUN pip install git+https://github.com/salesforce/pytorch-qrnn
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com torchtext
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com jieba gensim
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com opencc-python-reimplemented

ADD . /tmn

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r /tmn/requirements.txt

# Pytorch Preview 1.0.0
# RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

WORKDIR /tmn

# CMD sudo nvidia-docker run -it --rm -v /root/py/data:/data -p 11111:8888 -p 11112:6006 --ipc=host 10.202.107.19/sfai/tmn:q2q jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/tmn'
