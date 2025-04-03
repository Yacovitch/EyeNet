# Human Vision Based 3D Point Cloud Semantic Segmentation of Large-Scale Outdoor Scenes (CVPR PCV Workshop 2023)

This is the official GitHub page of **EyeNet** (CVPR PCV Workshop 2023, Oral and Poster presentation), an efficient and effective human vision-inspired 3d semantic segmentation network for point clouds. For more details, please refer to our paper ([CVPR PCV Workshop 2023](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Yoo_Human_Vision_Based_3D_Point_Cloud_Semantic_Segmentation_of_Large-Scale_CVPRW_2023_paper.html)).

### Preparation

- Clone this repository.

- There are two ways to prepare the environment. (However, it is recommended to use the docker container)

a. Using docker container.

You can install my docker container:

```
docker pull ausmlab/mrnet:cuda10.2-cudnn7-py3.7_pytorch1.5_TF2.9.1_automl
```

b. Setting up the environment on your own.

The code has been tested with Python 3.7, Tensorflow 11.1, Cuda 10.2, cuDNN 7.4.1 on Ubuntu 16.04.



- Create Conda Environment:

```
conda create -n eyenet python=3.5
source activate eyenet
```

- You need to update pip:

```
curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py
python get-pip.py
```

- Install Required Libraries and compile custom libraries:

```
pip install -r helper_requirements.txt
conda install -c conda-forge zip
sh compile_op.sh
conda install cudatoolkit=9.0
```

### Sensat Urban

- Download the SensatUrban Dataset from the official website (https://github.com/QingyongHu/SensatUrban).

- cambridge_block_0.ply and cambridge_block_1.ply contain less than 4mb of data, so they have to be removed before processing.

- Pre-processing dataset (Grid Sampling) by running:
```
python data_processing/input_preparation_Sensat.py --dataset_path "YOUR_DATA_PATH" --output_path "YOUR_OUTPUT_PATH"
```
Note: Grid size can be also adjusted for further details, please refer to the code.

- Start Training

```
python main_Sensat.py
```

Note: Before Training, please modify data_set_dir (line 21) to your sampled data directory in tool.py.


- Start Evaluation on validation set (for visualization):

```
python main_Sensat.py --mode val --model_path "YOUR_SAVED_MODEL"
```

Note: saved models are located in the "trained_weights/Sensat" folder.

Note: The "YOUR_SAVED_MODEL" path has to include snap-NumberofSteps e.g. trained_weights/Sensat/First_Train/snapshots/snap-17001.


- Start Evaluation on test set:

```
python main_Sensat.py --mode test --model_path "YOUR_SAVED_MODEL"
```

- The folder that contains the submission file will be saved in the "test" folder.
- This code will generate the submission file for submitting your result on the Online Server (https://codalab.lisn.upsaclay.fr/competitions/7113).



### DALES
- Downloading the LAS version of the DALES data set from the website https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php. Ply version does not include the number of return feature.

- Pre-processing dataset (Grid Sampling) by running:

```
python data_processing/input_preparation_DALES.py --dataset_path "YOUR_DATA_PATH" --output_path "YOUR_OUTPUT_PATH"
```


- Start Training:

```
python main_DALES.py
```
Note: Before Training, please modify data_set_dir (line 73) to your sampled data directory in tool.py.


- Start Evaluation:
```
python main_DALES.py --mode test --model_path "YOUR_SAVED_MODEL"
```
Note: saved models are located in the "trained_weights/DALES" folder.
Note: The "YOUR_SAVED_MODEL" path has to include snap-NumberofSteps e.g. trained_weights/DALES/First_Train/snapshots/snap-17001.

- The evaluation results will be saved in the "test" folder.


### Toronto3D
- If you have access to our Nas2 server, you can just download all dataset from NAS2/VM/jacob/data/Toronto3D
- Pre-processing dataset (Grid Sampling) by running:
```
python data_processing/input_preparation_toronto3D.py --dataset_path "YOUR_DATA_PATH" --output_path "YOUR_OUTPUT_PATH"
```

- Start Training:

```
python main_Toronto3D.py
```
Note: Before Training, please modify data_set_dir (line 125) to your sampled data directory in tool.py.


- Start Evaluation:
```
python main_Toronto3D.py --mode test --model_path "YOUR_SAVED_MODEL"
```
Note: saved models are located in the "trained_weights/Toronto3D" folder.
Note: The "YOUR_SAVED_MODEL" path has to include snap-NumberofSteps e.g. trained_weights/Toronto3D/First_Train/snapshots/snap-17001.

- The evaluation results will be saved in the "test" folder.

### Citation
Thank you for showing interest in our work. Please consider citing:
```
@InProceedings{Yoo_2023_CVPR,
    author    = {Yoo, Sunghwan and Jeong, Yeonjeong and Jameela, Maryam and Sohn, Gunho},
    title     = {Human Vision Based 3D Point Cloud Semantic Segmentation of Large-Scale Outdoor Scenes},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {6576-6585}
}
```


### Acknowledgement
Part of our work refers to ([nanoflann](https://github.com/jlblancoc/nanoflann)) and ([RandLA-Net](https://github.com/QingyongHu/RandLA-Net)).

### Updates
* 9/17/2023: Code uploaded!


### License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

