# cs-433-project-2-outliers
Resource-Efficient Machine Learning Algorithm Design for On-Implant Neurological Symptom Detection
 
In this project, we focus on developing an efficient Machine Learning model to process neural data in real time, with low power consumption, small on-chip area and fast inference.




## Installation
```shell
make install  # install the Python requirements and setup the paths
```
Download [trajectories from ECoG](https://drive.google.com/drive/folders/1DZC1ubNQzW-WndqRS7ZwRBGDofP2fSM3?usp=sharing), and the [pre-trained weights](https://drive.google.com/drive/folders/1-3C1Bt_H1_m98DUsWuLcTJG8oWbvETU4?usp=sharing). The folder structure should look like:
```
$drive
|-- data
`-- checkpoints
|   
`-- figures
```

The figures will be geberated by [run.py](https://github.com/CS-433/cs-433-project-2-outliers/blob/main/run.py) and be stored in drive/figures.

## Usage

### In all following cases you can use pe-trained weights by adding --pre-tarined=True to the command

### Baseline Model
For using the baseline model with no compression run:
```shell
python run.py
```
### Fixed Point Quantization
For applying Fixed Point Quantization run:
```shell
python run.py --fixed_pt_quantization=True
```
### Pruning
For applying Pruning run:
```shell
python run.py --pruning=True
```
### Trained Quantization and Weight Sharing:
For applying Trained Quantization and Weight Sharing run:
```shell
python run.py --trained_quantization=True
```


### Note that you cannot use Fixed Point Quantization, Prunning, and Trained Qunatization at the same time.



You can adjust the number of epochs for training in [run.py](https://github.com/CS-433/cs-433-project-2-outliers/blob/main/run.py) The training can be interupted by ctrl+c and the weights will be saved in checkpoints directory.

In [fpoint_quantization.ipynb](https://github.com/CS-433/cs-433-project-2-outliers/blob/main/fpoint_quantization.ipynb) you can reproduce the results published in the report for the Binarization and Fixed-Point Quantization method. 

In [pruning.ipynb]() you can reproduce the results published in the report for the Pruning method. 

In [trained_quantization.ipynb](https://github.com/CS-433/cs-433-project-2-outliers/blob/main/trained_quantization.ipynb) you can reproduce the results published in the report for the Trained Quantization and Weight Sharing method. Note that as kmeans clustering is not deterministic you might get slightly different esults that the ones in the report.


## Authors

Chabenat Eugénie : eugenie.chabenat@epfl.ch

Djambazovska Sara : sara.djambazovska@epfl.ch

Mamooler Sepideh : sepideh.mamooler@epfl.ch
