# Learner: 王振强
# Learn Time: 2022/4/8 19:32
# 图像宽高
height = 256  # 3472
width = 256   # 4624

learningRate = 3e-4  # 3e-4 #3.5e-4  # 1e-2
totalEpoch = 100

black_level = 1024   # gt 1006 16368 noise 1008 16383
white_level = 16383

batchSize = 16

# 五折交叉csv地址
train_fold_csv = './data/train_256_fold.csv'
# 训练图片文件夹路径
input_path = "./data/train/noisy_256/"
# 训练label文件夹路径
label_path = "./data/train/gt_256/"


