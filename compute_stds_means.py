import utils

#### compute stds and means
txt_path = "H:/DL-DATASET/360M/train.txt"
img_prefix = "H:/DL-DATASET/360M/images"
stds, means = utils.compute_std_mean(txt_path, img_prefix, NUM=1000)
print("stds = ", stds , "means = ", means)
