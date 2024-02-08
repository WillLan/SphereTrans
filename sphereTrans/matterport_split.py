import os
import glob

# 设置文件夹路径
file_name = '/media/lby/lby_8t/dataset/3d60/SunCG'
out_path = '/media/lby/lby_8t/dataset/3d60/suncg.txt'
count = 0
lens = 0
# 获取所有小文件夹的路径
    # 遍历根文件夹中的所有子文件夹

with open(out_path, 'w') as f:
    for folder_name in os.listdir(file_name):
        print(folder_name)
        d = folder_name.split('_')[1]
        print(d)
        if d=='color':
            f.write(folder_name + '\n')
#         folder_path = os.path.join(root_folder, folder_name)
#         # 确保路径是一个文件夹
#         if os.path.isdir(folder_path):
#             png_files = glob.glob(os.path.join(folder_path, '*.png'))
#             lens+=len(png_files)
#         # 将文件名写入txt文档
#             for png_file in png_files:
#             # 获取文件名（不包含路径）
#             # 写入txt文档
#                 txt_file.write(png_file + '\n')
#                 count+=1
#
# print(count)
# print(lens)
# print("文件名已成功写入txt文档。")