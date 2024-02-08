import os
import cv2

# def resize_and_save(input_folder, output, target_size):
#     # 创建输出文件夹（如果不存在）
#     if not os.path.exists(output):
#         os.makedirs(output)
#
#
#     for filename in os.listdir(input_folder):
#         # 构建输入和输出文件的完整路径
#         input_path = os.path.join(input_folder, filename)
#         print(filename)
#         img_name = filename.split('.')[0]
#         img_name = img_name.lstrip("0")
#         output_path = os.path.join(output, img_name+'.jpg')
#
#         # 读取图片
#         img = cv2.imread(input_path)
#         # 如果图片读取成功，进行尺寸变换
#         if img is not None:
#             # 将图片大小调整为目标尺寸
#             resized_img = cv2.resize(img, target_size)
#             # 保存变换后的图片
#             cv2.imwrite(output_path, resized_img)
#             print(f"Image {img_name} resized and saved.")
#
# # 设置输入和输出文件夹路径以及目标尺寸
# input_folder_path = '/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/CVIQD/resize_image'
# output_folder_path = '/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/CVIQD/image_resize'
# target_size = (1024, 512)  # 例如，设置为 (width, height)
#
# # 调用函数进行尺寸变换和保存
# resize_and_save(input_folder_path, output_folder_path, target_size)


def rename(input_folder, output, target_size):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output):
        os.makedirs(output)


    for filename in os.listdir(input_folder):
        # 构建输入和输出文件的完整路径
        input_path = os.path.join(input_folder, filename)
        print(filename)
        img_name = filename.split('.')[0]
        img_name = img_name[3:]
        output_path = os.path.join(output, img_name+'.jpg')

        # 读取图片
        img = cv2.imread(input_path)
        # 如果图片读取成功，进行尺寸变换
        if img is not None:
            # 将图片大小调整为目标尺寸
            # resized_img = cv2.resize(img, target_size)
            # 保存变换后的图片
            cv2.imwrite(output_path, img)
            print(f"Image {img_name} renamed and saved.")

# 设置输入和输出文件夹路径以及目标尺寸
input_folder_path = '/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/OIQA/resize_image'
output_folder_path = '/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/OIQA/image_resize'
target_size = (1024, 512)  # 例如，设置为 (width, height)

# 调用函数进行尺寸变换和保存
rename(input_folder_path, output_folder_path, target_size)