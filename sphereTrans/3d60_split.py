
file_name = '/media/lby/lby_8t/pano_seg/sphereTrans/sphereTrans/datasets/3d60_train.txt'
out_path = '/media/lby/lby_8t/dataset/3d60/matterport.txt'
rgb_depth_list = []
with open(file_name) as f:
    lines = f.readlines()
    for line in lines:
        rgb_depth_list.append(line.strip().split(" "))

print(len(rgb_depth_list))

with open(out_path, 'w') as f:
    for i in range(len(rgb_depth_list)):
        name = rgb_depth_list[i][0].split("/")[0]
        # print(name)
        if (name == 'Matterport3D'):
            f.write(rgb_depth_list[i][0] + ' ' + rgb_depth_list[i][1] + '\n')


