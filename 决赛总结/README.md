# 2022中兴捧月 决赛方案

### 1. 赛事背景

在中兴捧月大赛初赛中，举办方给出了200张training set图片，其中噪声生成方式采用人工合成的方法。真实Camera系统中噪声的标定一般需要在图像lab中拍摄灰阶图，离线标定噪声得到noise profile参数，进而在camera实时图像处理过程中，更好地去除噪声影响。

为了减轻繁杂的离线噪声标定工作，工程上提出了在线标定noise profile方法，期望通过实时camera raw图在线得到noise profile参数。

### 2. 赛事任务

赛题目的是需要学生提交在线noise profile估计算法。

为了方便同学们理解赛题内容，请在generateNoise.py文件denoise_raw()函数中，明确noise profile定义以及本次比赛噪声人工合成方法；

为了方便同学们理解评分规则，在./data/test/文件夹中，举办方提供了3张示例打分图片。在最终打分过程中，举办方会替换./data/test/中noisy图片以及生成不同的noise profile参数，以验证同学们所提出算法的泛化性能。

***\*学生需要在cal_noise_profile()函数中给出noise profile参数(a,b)的估计算法。\****

### 3. 评审规则

**Score = 100 - 100 * mean(evm)**

**evm = 0.5 * min(abs(a - gt_a)/gt_a ,1) + 0.5 * min(abs(b - gt_b)/gt_b, 1)**

其中a和b表示学生估计得到的noise profile参数值，gt_a和gt_b表示noise profile期望参数值，evm根据多张图片和多个noise profile参数得到，mean(.)表示均值，abs(.)表示绝对值。

学生提交的算法方案对于每张image运行时间需要小于2min，即算法嵌入generateNoise.py后，举办方本地PC运行时间小于2min。

作为时间参考，举办方本地PC直接运行generateNoise.py文件时间约为6s，举办方提出了一种简单的noise profile参数估计算法，合并入generateNoise.py中，本地PC运行generateNoise.py的时间约为10s。

***\*建议学生使用传统图像处理方案(即无需进行Deep learning training和inference过程)，以缩短完成比赛时长和满足运行时间要求。\****

评分过程中，noise profile参数范围是a=[1e-4,1e-2]，b=[1e-6,1e-3]。

评分过程中，black level=1024和white level=16383固定不变，学生无需刻意关注。

评分raw示例图见./data/test中图片。

学生允许调用open cv，tensorflow，sklearn等开源python包。

### 4. 解题思路

![](README_img\插图1.png)

![](README_img\插图2.png)

可以知道噪声可以估计的参数只有方差，但是有a和b两个参数,一张图统计方差是无法求解的，观察噪声图可知分辨率是比较高的，因而可以对原始图像不重叠切分若干张446×446大小切片，任取两张统计标准差sigma1,sigma2(最终选取噪声分布+δ和+2δ的均值代替标准差),同时两张切片图的均值raw1和raw2近似作为该区域所有像素点的raw值，由此可得上图中两个方程，由上面两个方程即可解出a,b值，不过这样得到的a,b值可信度不高,故迭代2000次随机抽取两张不重叠图像计算对应a,b值，最终取这2000组数据的中值作为最终所需a,b值，此时得到的a,b可信度非常高了。

### 5.决赛数据

数据与初赛的gt图相同,高斯噪声图由denoise_raw()接口产生。

### 6.参考文献

链接：https://pan.baidu.com/s/109WCMJI9K4wa7UaaXf4Thg   提取码：gse9