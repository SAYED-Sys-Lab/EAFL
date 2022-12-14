
## EAFL: Energy-Aware Federated Learning Framework on Battery-Powered Clients ([Paper](https://arxiv.org/abs/2208.04505))

This repository contains scripts and instructions of running EAFL, a framework for experimenting with federated learning battery-powered clients. 
EAFL extends [FedScale](https://fedscale.ai) framework which provides a diverse set of challenging and realistic benchmark datasets to facilitate scalable, comprehensive, 
and reproducible federated learning (FL) research. FedScale datasets are large-scale, encompassing a diverse range of important FL tasks, 
such as image classification, object detection, language modeling, speech recognition, and reinforcement learning. 
For each dataset, we provide a unified evaluation protocol using realistic data splits and evaluation metrics. 
To meet the pressing need for reproducing realistic FL at scale. The framework leverages an efficient evaluation platform for automated runtime to simplify and standardize the process of FL experimental setup and model evaluation. 
Our evaluation platform provides flexible APIs to implement new FL algorithms and include new execution backends with minimal developer efforts.  

***EAFL is open-source with permissive licenses and actively maintained, 
and we welcome feedback and contributions from the community!***

## Overview

* [Getting Started](#getting-started)
* [Realistic FL Datasets](#realistic-fl-datasets)
* [Run Experiments with FAR](#run-experiments-with-far)
* [Repo Structure](#repo-structure)
* [Note](#acknowledgements)
* [Contact](#contact)

## Getting Started 

Our ```install.sh``` will install the following automatically:

* Anaconda Package Manager
* CUDA 10.2

Note: if you prefer different versions of conda and CUDA, please check  comments in `install.sh` for details.

Run the following commands to install EAFL. 

```
git clone https://github.com/SAYED-Sys-Lab/EAFL
cd EAFL
source install.sh 
```

## Realistic FL Datasets

***We are adding more datasets! Please feel free to contribute!***

We provide real-world datasets for the federated learning community, and plan to release much more soon! Each is associated with its training, validation and testing dataset. A summary of statistics for training datasets can be found in Table, and you can refer to each folder for more details. Due to the super large scale of datasets, we are uploading these data and carefully validating their implementations to FAR. So we are actively making each dataset available for FAR experiments. 

CV tasks:

| Dataset       | Data Type   |# of Clients  | # of Samples   | Example Task | 
| -----------   | ----------- | -----------  |  ----------- |    ----------- |
| iNature       |   Image     |   2,295      |   193K        |   Classification |
| FMNIST        |   Image     |   3,400      |   640K        |   Classification  |    
| OpenImage     |   Image     |   13,771     |   1.3M        |   Classification, Object detection      |
| Google Landmark|  Image     |   43,484     |   3.6M        |   Classification       |
| Charades      |   Video     |    266       |   10K         |   Action recognition   |
| VLOG          |   Video     |    4,900     |   9.6k        |   Video classification, Object detection |

NLP tasks:

| Dataset       | Data Type   |# of Clients  | # of Samples   | Example Task | 
| -----------   | ----------- | -----------  |  ----------- |   ----------- |
| Europarl      |   Text      |   27,835     |   1.2M        |   Text translation  |
| Blog Corpus   |   Text      |   19,320     |   137M        |   Word prediction      |
| Stackoverflow |   Text      |   342,477    |   135M        |  Word prediction, classification |
| Reddit        |   Text      |  1,660,820   |   351M        |  Word prediction   |
| Amazon Review |   Text      | 1,822,925    |   166M        | Classification, Word prediction |
|  CoQA         |   Text      |     7,189    |   114K        |  Question Answering |
|LibriTTS       |   Text      |     2,456    |    37K        |   Text to speech    |
|Google Speech  |   Audio     |     2,618    |   105K        |   Speech recognition |
|Common Voice   |   Audio     |     12,976   |    1.1M       |   Speech recognition |

Misc Applications:

| Dataset       | Data Type   |# of Clients  | # of Samples   | Example Task | 
| -----------   | ----------- | -----------  |  ----------- |   ----------- |
|Taobao         |   Text      |     182,806  |    0.9M       |   Recommendation |
|Go dataset     |   Text      |     150,333  |    4.9M       |   Reinforcement learning | 

***Note that no details were kept of any of the participants age, gender, or location, and random ids were assigned to each individual. In using these datasets, we will strictly obey to their licenses, and these datasets provided in this repo should be used for research purpose only.***

Please go to `./dataset` directory and follow the dataset [README](dataset/README.md) for more details.

## Run Experiments
EAFL is an automated and easily-deployable evaluation platform, to simplify and standardize the FL experimental setup and model evaluation under a practical setting.

<img src="figures/faroverview.png" alt="Automated emulation enables the developer to benchmark various FL efforts with practical FL data and metrics">

Please go to `./core` directory and follow the core functions of the emulation platform [README](core/README.md) to set up FL training scripts.


## Repo Structure

```
Repo Root
|---- dataset     # Realistic datasets in EAFL
|---- core        # Experiment platform of EAFL
    |---- examples  # Examples of new plugins
    |---- evals     # Backend of job submission
    
```

## Notes
please consider to cite our paper if you use the code or data in your research project.

```bibtex
@inproceedings{eafl-arxiv,
  title={EAFL: Towards Energy-Aware Federated Learning on Battery-Powered Edge Devices},
  author={Amna Arouj and Ahmed M. Abdelmoniem},
  booktitle={arXiv:2208.04505},
  year={2021}
}
```

and  

```bibtex
@inproceedings{eafl-fededge22,
  title={Towards Energy-Aware Federated Learning on Battery-Powered Clients},
  author={Amna Arouj and Ahmed M. Abdelmoniem},
  booktitle={ACM FedEdge workshop - MobiCom},
  year={2022}
}
```

## Contact
Amna Arouj (amna.arouj@gmail.com), Ahmed M. Abdelmoniem (ahmedcs982@gmail.com)


