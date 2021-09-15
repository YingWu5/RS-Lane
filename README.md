# RS-Lane
The PyTorch implementation of the paper ["RS-Lane: A Robust Lane Detection Method Based on ResNeSt and Self-Attention Distillation for Challenging Traffic Situations"](https://www.hindawi.com/journals/jat/2021/7544355/)

## Basic Reults

### Results on TuSimple Benchmark
![Comparison on TuSimple testing set](https://github.com/YingWu5/RS-Lane/blob/master/images/tusimple.png)
![Results on TuSimple testing set](https://github.com/YingWu5/RS-Lane/blob/master/images/result1.png)

### Results on CULane Benchmark
![Comparison on CULane testing set](https://github.com/YingWu5/RS-Lane/blob/master/images/culane.png)
![Results on CULane testing set](https://github.com/YingWu5/RS-Lane/blob/master/images/result2.png)

## Get started
### 1. Prerequisites
- Python == 3.7
- PyTorch == 1.7
- CUDA == 10.2 (If you don't have CUDA, you can still train with CPU.)
- Other dependencies described in `requirements.txt`

### 2. Install
- Create a conda virtual environment and activate it.
```
conda create -n rslane python=3.7 -y
conda activate rslane
```
- Install Pytorch follow [the offical tutorials](https://pytorch.org/)
- Install other dependencies
```
pip install -r requirements.txt
```

### 3. Data preparation
- Download [TuSimple dataset](https://github.com/TuSimple/tusimple-benchmark/issues/3), and unzip the packs. The dataset structure should be as follows:
```
 tusimple
  `-- |-- test_set
      |   |-- clips
      |   `-- ...
      `-- train_set
          |-- clips
          |-- label_data_xxxx.json
          |-- label_data_xxxx.json
          |-- label_data_xxxx.json
          `-- ...
```
- Download [CULane dataset](https://xingangpan.github.io/projects/CULane.html), and unzip the packs. The dataset structure should be as follows:
```
$culane
|-- driver_100_30frame
|-- driver_161_90frame
|-- driver_182_30frame
|-- driver_193_90frame
|-- driver_23_30frame
|-- driver_37_30frame
|-- laneseg_label_w16
`-- list
```

### 4. Training
```
train(data_dir,ckpt_path,save_path,epoch_num,label)
python train_CULane.py \
    --data_dir /path/to/culane 
    --ckpt_path /path/to/checkpoint/file 
    --save_path /path/to/save/checkpoint/file
    --epoch_num 20
    --label <string>

python train_tusimple.py \
    --data_dir /path/to/tusimple 
    --ckpt_path /path/to/checkpoint/file 
    --save_path /path/to/save/checkpoint/file
    --epoch_num 50
    --label <string>
```
- Setup training set path using `--data_dir /path/to/dataset`, both training and validation data are loaded from this directory.
- Checkpoint file should be a `*.pth` file. If you are training from the begining, this one can be skiped.
- Setup the number of training epoch using `--epoch_num`.
- `--label <string>` will be included in the name of output and log files as a notice.
- After running train command, a directory `summary` should be created (if it does not already exists). The tensorboard summary details will be saved in it.


### 5. Testing
```
python test_CULane.py \
    --data_dir /path/to/culane 
    --ckpt_path /path/to/checkpoint/file 
    --save_path /path/to/save/output
    --label <string>

python test_tusimple.py \
    --data_dir /path/to/tusimple 
    --ckpt_path /path/to/checkpoint/file 
    --save_path /path/to/save/output
    --label <string>
```
- Setup testing set path using `--data_dir /path/to/dataset`.
- Checkpoint file should be a `*.pth` file.
- `--label <string>` will be included in the name of output directories and log files as a notice.
- Add `--show` to display output images while testing. In each iteration, after show images, the program pauses until a key is pressed.
- Add `--save` to save images into `/path/to/save/output` while testing.

### 6. Reproducing a result from the paper
Download checkpoint pth files.
- [GoogleDrive](https://drive.google.com/file/d/1Z9qSmkqJt6UQXILgAZoN6JdWQYe8XY2B/view?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1rTbD8S7x4xfYTsp9T742vw), 提取码: 0323

### 7. Evaluate
- Tusimple
```
cd tools
python evaluate.py --result /path/to/test/json --gt /path/to/groundtruth/json
```
- CULane
Using the [offical evaluate tools](https://github.com/XingangPan/SCNN).

## Citation
```
@article{zhang2021rs,
  title={Rs-Lane: a robust lane detection method based on ResNeSt and self-attention distillation for challenging traffic situations},
  author={Zhang, Ronghui and Wu, Yueying and Gou, Wanting and Chen, Junzhou},
  journal={Journal of advanced transportation},
  volume={2021},
  year={2021},
  publisher={Hindawi}
}
```
