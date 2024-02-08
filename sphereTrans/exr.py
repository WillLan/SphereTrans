
import os
def check_files_existence(txt_file, base_path):
    # 读取txt文件中的内容
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    line_num = 0
    # 检查每一对文件路径是否存在于指定路径中
    for line in lines:
        # 分割每一行的两个文件路径
        rgb_path, depth_path = line.strip().split()
        rgb_path, _ = recover_filename(rgb_path)
        depth_path, _ = recover_filename(depth_path)
        # 拼接成完整的文件路径
        full_rgb_path = os.path.join(base_path, rgb_path)
        full_depth_path = os.path.join(base_path, depth_path)
        line_num += 1
        # 检查文件是否存在
        if os.path.exists(full_rgb_path) and os.path.exists(full_depth_path):
            continue
            # print(f"Both files exist: {full_rgb_path} and {full_depth_path}")
        else:
            print(f"At least one file does not exist: {full_rgb_path} or {full_depth_path}")
            print(line_num)

def recover_filename(file_name):

    splits = file_name.split('.')
    rot_ang = splits[0].split('_')[-1]    # 获取旋转角度（180）
    file_name = splits[0][:-len(rot_ang)] + "0." + splits[-2] + "." + splits[-1]

    return file_name, int(rot_ang)    # Matterport3D/10_85cef4a4c3c244479c56e56d9a723ad21_color_0_Left_Down_0.0.png

# 替换为实际的txt文件路径和文件夹路径
txt_file_path = '/media/lby/lby_8t/pano_seg/sphereTrans/sphereTrans/datasets/3d60_train.txt'
base_folder_path = '/media/lby/lby_8t/dataset/3d60'

# 调用函数
check_files_existence(txt_file_path, base_folder_path)