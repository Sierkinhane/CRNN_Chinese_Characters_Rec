# Crnn_chinese_characters_rec
chinese characters recognition

## Descriptions in chinese：https://blog.csdn.net/Sierkinhane/article/details/82857572

## Test
There are images in test_images file, and you just run as follow.
> python3 test.py

## Train
before traning, you should prepare the dataset of characters.(described in csdn blog)

> python3 crnn_main.py --train_root (path of train lmdb dataset) --val_root (path of val lmdb dataset) --cuda(if have)


## 3.6 million chinese characters dataset：
![](https://github.com/Sierkinhane/LearningRecords/blob/master/chinese_char.png)

## Results：
![](https://github.com/Sierkinhane/crnn_chinese_characters_rec/blob/master/test_images/1.png)
![](https://github.com/Sierkinhane/crnn_chinese_characters_rec/blob/master/test_images/2.png)

## Training(accuray was 97.7% ultimately):
![](https://github.com/Sierkinhane/crnn_chinese_characters_rec/blob/master/test_images/3.png)

