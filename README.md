# LLM4CPTL: Large Language Models for Continuous Pedestrian Trajectory Learning

All scripts were originally developed and executed in a high performance computing (HPC) system named Lovelace. The system is hosted by the [Centros Nacionais de Processamento de Alto Desempenho de São Paulo (CENAPAD-SP)](https://www.cenapad.unicamp.br/), which is part of the [Sistema Nacional de Processamento de Alto Desempenho (SINAPAD)](https://www.lncc.br/sinapad/).

## Setup

1.  Install [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install);

2.  Activate conda for your bash session:
    ```
    source /opt/pub/spack/miniconda3/22.11.1/gcc/9.4.0/etc/profile.d/conda.sh
    ```

3.  Create and configure the conda environment:
    ```
    conda create -n cptlsgr python=3.6.13 -y
    conda activate cptlsgr
    ```
    3.1.  Packages instalation using conda:
     ```
      conda install numpy=1.19.5 pandas=1.1.5 matplotlib=3.3.4 pillow=8.2.0 tqdm=4.61.1 -c conda-forge -y
     ```
    3.2.  Packages instalation using pip:
     ```
     python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
     ```
     ```
     python -m pip install backcall==0.2.0 chardet==4.0.0 cycler==0.10.0 decorator==5.0.9 idna==2.10 ipython==7.16.1 \
        ipython-genutils==0.2.0 jedi==0.18.0 jsonpatch==1.32 jsonpointer==2.1 kiwisolver==1.3.1 parso==0.8.2 \
        pexpect==4.8.0 pickleshare==0.7.5 prompt-toolkit==3.0.18 ptyprocess==0.7.0 pygments==2.9.0 \
        pyparsing==2.4.7 pyzmq==22.1.0 scipy==1.5.4 torchfile==0.1.0 tornado==6.1 traitlets==4.3.3 \
        typing-extensions==3.10.0.0 urllib3==1.26.5 visdom==0.1.8.9 wcwidth==0.2.5 websocket-client==1.1.0 tensorboard==2.10.1
     ```
4. Create a folder for logs:
   ```bash
   mkdir cptl_with_social_gr/logs
   ```
