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

The figures will be geberated by [run.py]() and be stored in drive/figures.

## Usage

run.py consists of 4 parts:
1. The first part which concerns the main model.
2. BINARIZATON&FP_QUANTIZATION which peforms weight binarization and fixed point quantization.
3. PRUNING which applies the pruning method
4. TRAINED QUANTIZATION which applies the trained quantization and weight sharing method.

### Note that you cannot use Binaization, Prunning, and Trained Qunatization at the same time. So, when trying each of them make sure to comment the other two parts.

You can run run.py with pre-trained weights with the following command:
```shell
python run.py --pre-tarined=True 
```

If you want to train the models from scratch use the following command:

```shell
python run.py  
```

You can adjust the number of epochs for training in run.py. The training can be interupted by ctrl+c and the weights will be saved in checkpoints directory.

In [fpoint_quantization.ipynb]() you can reproduce the results published in the report for the Binarization and Fixed-Point Quantization method. 

In [pruning.ipynb]() you can reproduce the results published in the report for the Pruning method. 

In [trained_quantization.ipynb]() you can reproduce the results published in the report for the Trained Quantization and Weight Sharing method. 


## Authors

Chabenat Eugénie : eugenie.chabenat@epfl.ch

Djambazovska Sara : sara.djambazovka@epfl.ch

Mamooler Sepideh : sepideh.mamooler@epfl.ch
