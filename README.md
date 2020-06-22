# Characters Recognition

A Chinese characters recognition repository based on convolutional recurrent networks. 

## Performance

#### Recognize characters in pictures

<p align='center'>
<img src='images/demo.png' title='example' style='max-width:600px'></img>
</p>
<p align='center'>
<img src='images/demo_2.jpg' title='example2' style='max-width:600px'></img>
</p>

## Dev Environments
1. WIN 10 or Ubuntu 16.04
2. **PyTorch 1.2.0 (may fix ctc loss)** with cuda 10.0 🔥
3. yaml
4. easydict
5. tensorboardX

### Data
#### Synthetic Chinese String Dataset
1. Download the [dataset](https://pan.baidu.com/s/1ufYbnZAZ1q0AlK7yZ08cvQ)
2. Edit **lib/config/360CC_config.yaml** DATA:ROOT to you image path

```angular2html
    DATASET:
      ROOT: 'to/your/images/path'
```

3. Download the [labels](https://pan.baidu.com/s/1oOKFDt7t0Wg6ew2uZUN9xg) (password: eaqb)
4. Put *char_std_5990.txt* in **lib/dataset/txt/**
5. And put *train.txt* and *test.txt* in **lib/dataset/txt/**

    eg. test.txt
```
    20456343_4045240981.jpg 89 201 241 178 19 94 19 22 26 656
    20457281_3395886438.jpg 120 1061 2 376 78 249 272 272 120 1061
    ...
```
#### Or your own data
1. Edit **lib/config/OWN_config.yaml** DATA:ROOT to you image path
```angular2html
    DATASET:
      ROOT: 'to/your/images/path'
```
2. And put your *train_own.txt* and *test_own.txt* in **lib/dataset/txt/**

    eg. test_own.txt
```
    20456343_4045240981.jpg 你好啊！祖国！
    20457281_3395886438.jpg 晚安啊！世界！
    ...
```
**note**: fixed-length training is supported. yet you can modify dataloader to support random length training.   

## Train
```angular2html
   [run] python train.py --cfg lib/config/360CC_config.yaml
or [run] python train.py --cfg lib/config/OWN_config.yaml
```
```
#### loss curve

```angular2html
   [run] cd output/360CC/crnn/xxxx-xx-xx-xx-xx/
   [run] tensorboard --logdir log
```

#### loss overview(first epoch)
<center/>
<img src='images/train_loss.png' title='loss1' style='max-width:800px'></img>
</center>
<p>
<img src='images/tb_loss.png' title='loss1' style='max-width:600px'></img>
</p>

## Demo
```angular2html
   [run] python demo.py --image_path images/test.png --checkpoints output/checkpoints/mixed_second_finetune_acc_97P7.pth
```
## References
- https://github.com/meijieru/crnn.pytorch
- https://github.com/HRNet



