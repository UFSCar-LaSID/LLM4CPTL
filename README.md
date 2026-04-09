# LLM4CPTL: Large Language Models for Continuous Pedestrian Trajectory Learning

## Setup

1.  Have [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) installed;

2.  Activate conda for your bash session, e.g., for Miniconda:
    ```bash
    source /opt/pub/spack/miniconda3/22.11.1/gcc/9.4.0/etc/profile.d/conda.sh
    ```

3.  Create and activate the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate cptlsgr38
    ```

4. Create a folder for logs:
    ```bash
    mkdir logs
    ```

5. Make sure the `.sh` files have execution permission:
    ```bash
    chmod +x bash_scripts/*.sh
    ```

6. Pre-processing:
    ```bash
    ./bash_scripts/preprocessing_mllm_image2text.sh
    ./bash_scripts/preprocessing_mllm_text2embedding.sh
    ```

7. Experiments:

    7.1 Individual learning (IL):
    ```bash
    ./bash_scripts/experiment_IL_ETH_ETH.sh
    ./bash_scripts/experiment_IL_UCY_UCY.sh
    ./bash_scripts/experiment_IL_inD_inD.sh
    ./bash_scripts/experiment_IL_INTERACTION_INTERACTION.sh
    ```

    7.2 Baselines:
    ```bash
    ./bash_scripts/experiment_CLNR.sh
    ./bash_scripts/experiment_CLSGR.sh
    ```

    7.3 Our method:
    ```bash
    ./bash_scripts/experiment_ours.sh
    ```