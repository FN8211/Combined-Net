import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

start_time = time.time()

colors = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
            (128, 64, 12)]

class_number = len(colors)

"""
[ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ]
"""
colors = [(color[2], color[1], color[0]) for color in colors]      #OpenCV读取图像后，图像会转化为BGR格式


# 假设你有两个文件夹，一个包含txt文件，另一个包含掩膜文件
txt_folder = 'txt_faster_rcnn'  # 替换为你的txt文件所在的文件夹路径
mask_folder = 'mask'  # 替换为你的掩膜文件所在的文件夹路径

# 获取两个文件夹中所有文件的列表
txt_files = [os.path.join(txt_folder, f) for f in os.listdir(txt_folder) if f.endswith('.txt')]
mask_files = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith(('.png'))]

def fill_center_percentage(image_path, box, new_color, percentage=0.7):   #在标记框中完全没有掩膜时生成一定类别和面积的掩膜  v4
    #print(box, new_color)            #####
    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 计算标记框的中心点
    center_x = (box[0] + box[2]) // 2
    center_y = (box[1] + box[3]) // 2

    # 计算中心区域的宽度和高度
    width = box[2] - box[0]
    height = box[3] - box[1]
    center_width = int(width * (percentage ** 0.5))
    center_height = int(height * (percentage ** 0.5))

    # 计算中心区域的左上角和右下角坐标
    left = center_x - center_width // 2
    top = center_y - center_height // 2
    right = center_x + center_width // 2
    bottom = center_y + center_height // 2

    # 确保坐标不会超出标记框的边界
    left = max(left, box[0])
    top = max(top, box[1])
    right = min(right, box[2])
    bottom = min(bottom, box[3])

    # 如果计算出的中心区域有效（即不是空的），则填充颜色
    if left < right and top < bottom:
        draw.rectangle([left, top, right, bottom], fill=new_color)

    # 保存修改后的图像
    image.save(image_path)

# 遍历匹配的文件
for file_path in txt_files:
    start_time = time.time()
    # 提取不带扩展名的文件名
    file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    print(file_name_without_ext)
    # 查找对应的掩膜文件
    image_path = next((f for f in mask_files if os.path.splitext(os.path.basename(f))[0] == file_name_without_ext), None)
    # 读取图像
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    # 计算图像面积
    image_area = height * width
    # 打印图像面积
    #print(f"The image area is: {image_area} pixels")

    # 确保图像读取成功
    if image is None:
        print("Error: Could not read the image.")
        exit()

    # 初始化一个列表来存储所有颜色的轮廓
    all_contours = []
    # 创建一个空字典来存储不同类别的标记框位置信息
    bounding_boxes = {}
    # 创建一个外层列表，长度为21，每一项初始化为一个空列表
    bounding_boxes_list = [[] for _ in range(class_number)]

    #读取txt文件中的标记框
    with open(file_path, 'r') as file:

        if not file.readline():           #当没有完全没有检测框时跳过对掩膜的处理，因为因为目标识别没检测到物体但语义分割检测到物体了  v2
            print(f"{file_name_without_ext} has been skipped")
            continue
        file.seek(0)

        # 逐行读取文件内容
        for line in file:
            # 去除行尾的换行符，并分割每行内容
            parts = line.strip().split()

            # 检查是否至少有5个部分（类别和四个坐标）
            if len(parts) >= 5:
                # 将类别转换为整数
                category = int(parts[0])

                # 解析坐标（将字符串转换为整数）
                x1, y1, x2, y2 = map(int, parts[1:])

                # 创建一个元组来表示标记框的位置
                box = (x1, y1, x2, y2)

                # 如果该类别尚未在字典中，则添加它并设置其值为一个空列表
                if category not in bounding_boxes:
                    bounding_boxes[category] = []

                # 将标记框位置添加到对应类别的列表中
                bounding_boxes[category].append(box)
                bounding_boxes_list[category].append(5)

    #将掩膜中各种颜色的轮廓储存起来
    # 遍历每种颜色
    for color in colors:
        # 创建一个与原始图像相同大小的掩膜，用于提取当前颜色
        mask = np.all(image == color, axis=-1)

        # 查找掩膜中的轮廓
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检查contour是否不是空列表（即是否有值）
        #print(color)

        if (color == (0, 128, 192)) or (color == (0, 192, 0)):  # 跳过对chair. diningtable和sofa的处理  v2
            contours = []
            #print("skip color")


        # 将找到的轮廓添加到列表中
        all_contours.append(contours)
    # 现在，all_contours列表包含了所有20种颜色的轮廓
    # 每个轮廓列表对应一种颜色



    contour_person = all_contours[14] #额外保留person的轮廓

    #获取要处理的轮廓
    count_class = 0
    empty_box_contour = []
    for count_class in range(class_number):
        if not all_contours[count_class]:  # 没轮廓有类别
            empty_box_contour.append(None)
            #print(f"{count_class + 1} skip for empty")
            continue
        if count_class not in bounding_boxes and all_contours[count_class]:  # 有轮廓没标记框
            for index, large_contour in enumerate(all_contours[count_class]):
                contour_area = cv2.contourArea(large_contour)
                print(f"A contour in {index} place has area of {contour_area}")

                # 检查轮廓面积是否大于图像面积的70%           v3.1
                if contour_area > 0.25 * image_area:
                    # 如果超过了阈值，则提示
                    print(f"A contour in {index} place has area of {contour_area} which covers more than 25% of the image.")
                    before_element = all_contours[count_class][:index]
                    # 第二个切片包含要删除元素之后的所有元素
                    after_element = all_contours[count_class][index + 1:]
                    # 连接两个切片来创建新的元组
                    new_tuple = before_element + after_element
                    # 用新的元组替换列表中的原始元组
                    all_contours[count_class] = new_tuple



                if count_class == 12:                      #horse类有轮廓但目标检测没识别到         v3.2
                    contour_horse = all_contours[12]
                    for contour1 in contour_person:
                        contour1_area = cv2.contourArea(contour1)
                        print(f"contour percentage of person is {contour1_area / image_area * 100}")
                        if (contour1_area / image_area * 100) < 1:
                            continue
                        x1, y1, w1, h1 = cv2.boundingRect(contour1)
                        for index, contour2 in enumerate(contour_horse):
                            contour2_area = cv2.contourArea(contour2)
                            x2, y2, w2, h2 = cv2.boundingRect(contour2)
                            # 检查两个矩形是否相邻或重叠
                            if (x1 < x2 + w2 and x1 + w1 > x2) or (y1 < y2 + h2 and y1 + h1 > y2):
                                print(f"contour percentage of horse is {contour2_area / image_area * 100}")
                                if (contour2_area / image_area * 100) < 1:
                                    continue
                                print(contour1_area, contour2_area)
                                before_element = all_contours[12][:index]
                                # 第二个切片包含要删除元素之后的所有元素
                                after_element = all_contours[12][index + 1:]
                                # 连接两个切片来创建新的元组
                                new_tuple = before_element + after_element
                                # 用新的元组替换列表中的原始元组
                                all_contours[12] = new_tuple





            empty_box_contour.append(all_contours[count_class])
            # print(empty_box_contour[count_class])                                   ####################
            # print(type(empty_box_contour))
            # print(f"store contours of {count_class + 1} for empty bounding box")
            continue
        empty_box_contour.append(None)
        outside_points = []
        for contour in all_contours[count_class]:
            # 遍历轮廓中的每个点
            # print(f"length of contour is {len(contour)}")
            for point in contour:
                # print(point)
                # 获取点的坐标
                x, y = point[0]
                #print(bounding_boxes)
                for index1 in bounding_boxes:
                    for index2, box in enumerate(bounding_boxes[index1]):
                        #print(f"box are {box}")
                        
                        x1, y1, x2, y2 = box
                        # 检查点是否在边界框内
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            if bounding_boxes_list[index1][index2] > 0:
                                bounding_boxes_list[index1][index2] = bounding_boxes_list[index1][index2] - 1
                # 假设 count_class 是有效的，并且 bounding_boxes[count_class] 包含边界框列表
                pixel_in_any_box = False
                for box in bounding_boxes[count_class]:
                    # 边界框应该是一个包含四个坐标的列表或元组 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box
                    # 检查点是否在边界框内
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        pixel_in_any_box = True
                        """
                        if bounding_boxes_list[count_class][index] > 0:
                            bounding_boxes_list[count_class][index] = bounding_boxes_list[count_class][index] - 1
                        """
                        break  # 如果点在一个边界框内，则不需要检查其他边界框

                # 如果点不在任何边界框内，则打印消息
                if not pixel_in_any_box:
                    # outside_points.append(point)
                    outside_points.append([x, y])
                    # print(f"Point ({x}, {y}) is outside all defined regions of class {count_class + 1}.")
        if outside_points:
            height, width, channels = image.shape
            # 创建一个全黑的图像
            image_original = np.zeros((height, width), dtype=np.uint8)

            # 在图像上根据outside_point绘制白色点
            for point in outside_points:
                x, y = point
                image_original[y, x] = 255
            # 创建一个与原始图像相同大小的掩膜，用于提取当前颜色（在这个例子中是白色点）
            mask = image_original == 255
            # 查找掩膜中的轮廓
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            empty_box_contour.append(contours)


    #查找完全没有轮廓经过的标记框        v4
    for classes in range(class_number):
        if not bounding_boxes_list[classes]:
            continue
        else:
            for index in range(len(bounding_boxes_list[classes])):
                if bounding_boxes_list[classes][index] > 0:
                    new_color = colors[classes]
                    new_color = new_color[2], new_color[1], new_color[0]  # OpenCV读取图像后，图像会转化为BGR格式
                    box = bounding_boxes[classes][index]
                    fill_center_percentage(image_path, box, new_color)


    #判断轮廓需要改变的颜色
    #print(f"length of empty_box_contour {len(empty_box_contour)}")
    verify_point_num = 20
    image = cv2.imread(image_path)
    for j in range(class_number):
        if empty_box_contour[j] is None:
            #print(f"轮廓 {colors[j]} 在empty_box_contour中是None")
            continue
        else:
            #print(f"颜色 {colors[j]} 在empty_box_contour中")
            for change_color_contour in empty_box_contour[j]:
                random_points = []
                select_color = (0, 0, 0)
                if len(change_color_contour) >= verify_point_num:
                    #print("length of change_color_contour larger than or equal to 20")
                    for boundary in range(verify_point_num):
                        random_index = np.random.randint(0, len(change_color_contour))
                        random_point = change_color_contour[random_index][0]  # 注意：contour[i]是一个包含两个元素的元组，(x, y)
                        # 将随机选择的点添加到列表中
                        random_points.append(random_point)
                else:
                    #print("length of change_color_contour smaller than 20")
                    for point in change_color_contour:
                        random_points.append(point[0])
                for point in random_points:
                    x, y = point  # 假设point是一个包含两个元素的元组或列表
                    convert_to_black = True

                    # 检查点是否在任何其他类别的边界框内
                    for k in range(class_number):
                        if k == j:
                            continue
                        if k not in bounding_boxes:
                            continue
                        for box in bounding_boxes[k]:
                            x1, y1, x2, y2 = box  # 假设边界框是一个包含四个元素的列表或元组
                            # 检查点是否在边界框内
                            if (x1 <= x <= x2).all() and (y1 <= y <= y2).all():
                                select_color = colors[k]
                                # print(f"检验点 {point} 位于 {colors[k]} 中")
                                convert_to_black = False
                                break

                    # 如果点不在任何边界框内，则将其绘制为黑色
                    if convert_to_black:
                        select_color = (0, 0, 0)
                        # print(f"检验点 {point} 位于黑色中")
                #print(f"change contour of color {colors[j]} to {select_color}")
                cv2.fillPoly(image, pts=[change_color_contour], color=select_color)
    cv2.imwrite(image_path, image)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")


