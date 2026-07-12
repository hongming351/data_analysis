# Letter Recognition — 字母识别数据集

> Database of character image features; try to identify the letter  
> 字符图像特征数据库；尝试识别字母

## 数据集概览

| 属性 | 说明 |
|------|------|
| **Dataset Characteristics** | Multivariate / 多元 |
| **Subject Area** | Computer Science / 计算机科学 |
| **Associated Tasks** | Classification / 分类 |
| **Feature Type** | Integer / 整数 |
| **# Instances** | 20,000 |
| **# Features** | 16 |
| **Has Missing Values?** | No / 无 |

---

## 数据集描述

The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital letters in the English alphabet. The character images were based on **20 different fonts** and each letter within these 20 fonts was randomly distorted to produce a file of **20,000 unique stimuli**. Each stimulus was converted into **16 primitive numerical attributes** (statistical moments and edge counts) which were then scaled to fit into a range of integer values from **0 through 15**.

We typically train on the first **16,000 items** and then use the resulting model to predict the letter category for the remaining **4,000**.

> 目标是识别大量黑白矩形像素显示器中的每一个作为英文字母表中的 26 个大写字母之一。字符图像基于 **20 种不同的字体**，并且这 20 种字体中的每个字母都被随机扭曲，以生成一个包含 **20,000 个独特刺激** 的文件。每个刺激被转换为 **16 个原始数值属性**（统计矩和边缘计数），然后被缩放到 **0 到 15** 的整数范围内。我们通常在前 **16,000 项** 上进行训练，然后使用生成的模型来预测剩余 **4,000 项** 的字母类别。

---

## 变量表 (Variables Table)

| # | Column Name | CSV 列名 | Role | Type | Description | 描述 |
|---|------------|----------|------|------|-------------|------|
| 1 | lettr | letter | **Target** | Categorical | capital letter (A–Z) | 大写字母 |
| 2 | x-box | x_box | Feature | Integer | horizontal position of box | 框的水平位置 |
| 3 | y-box | y_box | Feature | Integer | vertical position of box | 框的垂直位置 |
| 4 | width | width | Feature | Integer | width of box | 框的宽度 |
| 5 | high | high | Feature | Integer | height of box | 框的高度 |
| 6 | onpix | onpix | Feature | Integer | total # on pixels | 像素总数 |
| 7 | x-bar | x_bar | Feature | Integer | mean x of on pixels in box | 框内像素 x 的平均值 |
| 8 | y-bar | y_bar | Feature | Integer | mean y of on pixels in box | 框内像素 y 的平均值 |
| 9 | x2bar | x2bar | Feature | Integer | mean x variance | x 的方差均值 |
| 10 | y2bar | y2bar | Feature | Integer | mean y variance | y 的方差均值 |
| 11 | xybar | xybar | Feature | Integer | mean x y correlation | x 和 y 的相关系数均值 |
| 12 | x2ybr | x2ybr | Feature | Integer | mean of x * x * y | x*x*y 的均值 |
| 13 | xy2br | xy2br | Feature | Integer | mean of x * y * y | x*y*y 的均值 |
| 14 | x-ege | x_ege | Feature | Integer | mean edge count left to right | 从左到右的平均边缘计数 |
| 15 | xegvy | xegvy | Feature | Integer | correlation of x-ege with y | x-ege 与 y 的相关性 |
| 16 | y-ege | y_ege | Feature | Integer | mean edge count bottom to top | 从下到上的平均边缘计数 |
| 17 | yegvx | yegvx | Feature | Integer | correlation of y-ege with x | y-ege 与 x 的相关性 |

---

## CSV 文件对应信息

- **文件名**: letter-recognition.csv
- **总行数**: 20,000（含表头）
- **总列数**: 17（1 个目标列 + 16 个特征列）
- **所有值均为整数**，范围 0–15（letter 列除外，为大写字母 A–Z）
