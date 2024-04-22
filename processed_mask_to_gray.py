import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 定义要检查的颜色列表和颜色容忍度
colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
          (128, 64, 128)]
tolerance = 20

# 输入和输出文件夹路径
input_folder = 'mask'  # 替换为你的输入文件夹路径
output_folder = 'temp_mask'  # 替换为你的输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):  # 根据需要添加更多格式
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        color_image = cv2.imread(image_path)

        # 创建与原始图像相同大小的单通道灰度图像，并初始化为0
        gray_image = np.zeros_like(color_image[:, :, 0], dtype=np.uint8)

        # 遍历原始图像的每个像素
        for y in range(color_image.shape[0]):
            for x in range(color_image.shape[1]):
                # 获取当前像素的RGB值
                b, g, r = color_image[y, x]

                # 检查像素值是否在指定颜色的容忍度范围内
                for color in colors:
                    color_r, color_g, color_b = color
                    if (r >= color_r - tolerance and r <= color_r + tolerance) and \
                            (g >= color_g - tolerance and g <= color_g + tolerance) and \
                            (b >= color_b - tolerance and b <= color_b + tolerance):
                        # 如果是，则设置灰度图像的相应像素为颜色的位数
                        gray_image[y, x] = colors.index(color)  # 颜色的位数，这里假设颜色值的范围是0-255
                        break  # 找到匹配的颜色后，跳出内层循环

        # 保存处理后的图像到输出文件夹
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, gray_image)
        print(filename)

print("处理完成，结果已保存到:", output_folder)