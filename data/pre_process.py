import os
import csv
import cv2
import json
import argparse

######################################################################
# Options
# --------
parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str,
                    default='../../datas/openpose_output/jsons/')
parser.add_argument('--org_img_path', type=str,
                    default='../../datas/images/')
parser.add_argument('--rearg_img_path', type=str,
                    default='../../datas/images/rearrange/')
parser.add_argument('--pose_img_path', type=str,
                    default='../../datas/openpose_output/images/')
parser.add_argument('--pose2_img_path', type=str,
                    default='../../datas/openpose_output/images2/')
parser.add_argument('--edit_img_path', type=str,
                    default= '../../datas/images_edit/')
parser.add_argument('--dataset', type=str, default='duke',
                    choices=['market', 'duke', 'all'])
parser.add_argument('--subset', type=str, default='gallery',
                    choices=['query', 'gallery', 'train', ''])
parser.add_argument('--pose', type=bool, default=True)
args = parser.parse_args()

if args.subset == 'train':
    sub_folder = '/bounding_box_train/'
elif args.subset == 'gallery':
    sub_folder = '/bounding_box_test/'
elif args.subset == 'query':
    sub_folder = '/query/'
else:
    sub_folder = '/'

if args.dataset == 'all':
    under_line = ''
else:
    under_line = '_'


def rearrange(path, out):
    images = os.listdir(path)

    total_num = 0

    for img in images:
        print(img)
        img_split = img.split('_')
        class_name = img_split[0]
        class_path = args.rearg_img_path + args.dataset + sub_folder + class_name + '/'

        if not os.path.exists(class_path):
            os.makedirs(class_path)

        image = cv2.imread(path + img)
        cv2.imwrite(class_path + img, image)

        total_num += 1

    print('Number of images:', len(images))
    print('Total number of imgs:', total_num)


def counting(path):
    folders = os.listdir(path)
    print(len(folders))
    num_list =[]
    csvfile = args.org_img_path + args.dataset + '_' + args.subset + '.csv'
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')

        for class_fold in folders:
            img_nums = len(os.listdir(path + class_fold + '/'))
            num_list.append(img_nums)
            writer.writerow([class_fold, img_nums])

    print('max img nums:', max(num_list))
    print('min img nums:', min(num_list))


def flip_img(path):
    classes = os.listdir(path)
    print('classes: ', len(classes))

    for index, name in enumerate(classes):
        # print(index)
        class_path = path + name + "/"
        for img_name in os.listdir(class_path):
            print(img_name)
            img_name_split = img_name.split('_')
            real_id = img_name_split[0]
            # print(real_id)

            img_name_split2 = img_name.split('.')
            only_name = img_name_split2[0]
            # print(only_name)

            img_path = class_path + img_name
            image = cv2.imread(img_path)
            pic = cv2.resize(image, (128, 384), interpolation=cv2.INTER_CUBIC)
            flipped = cv2.flip(pic, 1)

            class_floder = folder_n + real_id + '/'

            if not os.path.exists(class_floder):
                os.makedirs(class_floder)

            cv2.imwrite(class_floder + img_name, pic)
            cv2.imwrite(class_floder + only_name+'_flip.jpg', flipped)
            # cv2.imshow("Image", pic)
            # cv2.imshow("Image", flipped)


def resize_img():
    path = dataset_path + market + test
    new_folder = folder_n + market + test
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for img_name in os.listdir(path):
        print(img_name)
        img_path = path + img_name
        image = cv2.imread(img_path)
        pic = cv2.resize(image, (128, 384), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(new_folder + img_name, pic)


def rename_image(path, save_path):
    total_num = 0
    for img_name in os.listdir(path):
        # print(img_name)
        img_name_split = img_name.split('_')
        real_id = int(img_name_split[0])
        cam_num = int(img_name_split[1])
        # print('cam_num:', cam_num)

        img_path = path + img_name
        image = cv2.imread(img_path)
        pic = cv2.resize(image, (128, 384), interpolation=cv2.INTER_CUBIC)
        new_name = str(real_id).zfill(4) + '_' + str(cam_num).zfill(2)+'_'+ str(total_num).zfill(6)+'.jpg'
        print(new_name)
        cv2.imwrite(save_path+new_name, pic)
        total_num += 1


def trans_json(json_path, save_file):
    points_dic = {}

    for json_name in os.listdir(json_path):
        print(json_name)
        img_name_split = json_name.split('_')
        org_name = img_name_split[0]+'_'+img_name_split[1]+'_'+img_name_split[2]

        point_list = []

        with open(json_path+json_name) as json_file:
            config = json.loads(json_file.read())
            if config['people'] == []:
                points_dic[org_name] = point_list
            else:
                points = config['people'][0]['pose_keypoints_2d']
                for i in range(25):
                    point_list.append(points[i * 3:(i + 1) * 3])
                points_dic[org_name] = point_list
            # print(len(point_list))
            # print(point_list)

    print(len(points_dic))

    with open(save_file, 'w') as fp:
        json.dump(points_dic, fp, sort_keys=True, indent=4)



read_path = args.edit_img_path + args.dataset + sub_folder
out_path = args.org_img_path + args.dataset + sub_folder

rearrange(out_path, out_path)


read_path = args.rearg_img_path + args.dataset + sub_folder

counting(read_path)

# read_path = args.json_path + args.dataset + sub_folder
# save_file = args.json_path + args.dataset + under_line + args.subset + '.json'

# trans_json(read_path, save_file)


# if args.pose == True:
#     img_path = args.pose_img_path + args.dataset + sub_folder
# else:
#     img_path = args.org_img_path + args.dataset + sub_folder
#
# json_file = args.json_path + args.dataset + args.subset + '.json'
#
# save_path = args.pose2_img_path + args.dataset + sub_folder

# flip_img(path)
# resize_img()
# draw_points()
# draw_points2()









