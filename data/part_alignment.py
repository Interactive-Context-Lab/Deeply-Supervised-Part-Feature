import os
# import tensorflow as tf
import cv2
import json
import argparse
import numpy as np

######################################################################
# Options
# --------
parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str,
                    default= '../../datas/openpose_output/jsons/')  # 原本OpenPose輸出的骨架json檔路徑
parser.add_argument('--org_img_path', type=str,
                    default='../../datas/images/')                  # 原本資料集路徑
parser.add_argument('--pose_img_path', type=str,
                    default='../../datas/openpose_output/images/')
parser.add_argument('--pose2_img_path', type=str,
                    default='../../datas/openpose_output/images2/')
parser.add_argument('--edit_img_path', type=str,
                    default='../../datas/images_edit/')             # 對齊後輸出圖像路徑
parser.add_argument('--dataset', type=str, default='duke',
                    choices=['market', 'duke', 'all'])
parser.add_argument('--subset', type=str, default='train',
                    choices=['query', 'gallery', 'train', ''])
parser.add_argument('--pose', type=bool, default=True)
args = parser.parse_args()


def draw_points(img_path, json_file, save_path):
    # 18 points
    # colors = [[0, 0, 255], [0, 85, 255],
    #           [0, 170, 255], [0, 255, 255], [0, 255, 170],
    #           [0, 255, 85], [0, 255, 0], [85, 255, 0],
    #           [170, 255, 0], [255, 255, 0], [255, 170, 0],
    #           [255, 85, 0], [255, 0, 0], [255, 0, 85],
    #           [255, 0, 170], [255, 0, 255], [170, 0, 255], [85, 0, 255]]

    # 25 points
    colors = [[0, 0, 255], [0, 85, 255],  # 0(nose), 1(neck)
              [0, 255, 255], [0, 255, 170], [0, 255, 85], # 2(RShoulder), 3(RElbow), 4(RWrist)
              [0, 255, 0], [85, 255, 0], [170, 255, 0], #5(LShoulder), 6(LElbow), 7(LWrist)
              [0, 170, 255], # 8(MidHip)
              [255, 255, 0], [255, 170, 0], [255, 85, 0], # 9(RHip), 10(RKnee), 11(RAnkle)
              [255, 0, 0], [255, 0, 85], [255, 0, 170], # 12(LHip), 13(LKnee), 14(LAnkle)
              [255, 0, 255], [170, 0, 255], [85, 0, 255], [45, 0, 255], # 15(REye), 16(LEye), 17(REar), 18(LEar)
              [255, 0, 190], [255, 0, 210], [255, 0, 230], # 19(LBigToe), 20(LSmallToe), 21(LHeel)
              [255, 65, 0], [255, 45, 0], [255, 25, 0]] # 22(RBigToe), 23(RSmallToe), 24(RHeel)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(json_file) as json_f:
        config = json.loads(json_f.read())

        for img_name in os.listdir(img_path):
            print(img_name)
            img_name_split = img_name.split('_')
            org_name = img_name_split[0]+'_'+img_name_split[1]+'_'+img_name_split[2]

            points = config[org_name]

            image = cv2.imread(img_path + img_name)

            if points == []:
                cv2.imwrite(save_path + img_name, image)
                continue

            for i in range(25):
                # print(i)
                if points[i][2] == 0:
                    continue
                # print(tuple([int(points[i][0]), int(points[i][1])]))
                image = cv2.circle(image, tuple([int(points[i][0]), int(points[i][1])]), 5, colors[i], thickness=-1)

            cv2.imwrite(save_path + img_name, image)


def part_aligned(img_path, json_file, align_save_path1, align_save_path2, align_save_path3, align_save_path4):

    if not os.path.exists(align_save_path1):
        os.makedirs(align_save_path1)
    if not os.path.exists(align_save_path2):
        os.makedirs(align_save_path2)
    if not os.path.exists(align_save_path3):
        os.makedirs(align_save_path3)
    if not os.path.exists(align_save_path4):
        os.makedirs(align_save_path4)

    with open(json_file) as json_f:
        config = json.loads(json_f.read())

        for img_name in os.listdir(img_path):
            print(img_name)
            img_name_split = img_name.split('_')
            org_name = img_name_split[0] + '_' + img_name_split[1] + '_' + os.path.splitext(img_name_split[2])[0]

            points = config[org_name]

            image = cv2.imread(img_path + img_name)
            print('input-', np.shape(image))

            if points == []:
                cv2.imwrite(align_save_path4 + img_name, image)
                continue

            x_min = 64
            x_max = 64

            joints =[]
            point3 = points[3][1]
            point6 = points[6][1]
            point9 = points[10][1]
            point12 = points[13][1]
            top_ap = 0
            bottom_ap = 0
            min_ap = 0
            max_ap = 0
            # top = 0
            # bottom = 384

            for i in range(25):
                # print(i)
                if points[i][2] == 0:
                    continue
                joints.append(i)

                if int(points[i][0]) < x_min:
                    x_min = int(points[i][0])
                if int(points[i][0]) > x_max:
                    x_max = int(points[i][0])

            if len(joints) <= 10:
                cv2.imwrite(align_save_path4 + img_name, image)
                continue


            # if 1 in joints and 8 in joints:
            #     part_h = points[1][1]

            if 3 in joints and 10 in joints:
                part_h = point9 - point3
                top = point3 - part_h
                bottom = point9 + part_h
            elif 6 in joints and 13 in joints:
                part_h = point12 - point6
                top = point6 - part_h
                bottom = point12 + part_h
            elif 3 in joints and 13 in joints:
                part_h = point12 - point3
                top = point3 - part_h
                bottom = point12 + part_h
            elif 6 in joints and 10 in joints:
                part_h = point9 - point6
                top = point6 - part_h
                bottom = point9 + part_h
            elif 9 in joints:
                if 1 in joints or 2 in joints or 5 in joints:
                    top = points[8][1] - 192
                else:
                    top = points[8][1]
                    top_ap = points[8][1] - 192

                if 11 in joints or 14 in joints:
                    bottom = points[8][1] + 192
                elif 10 in joints or 13 in joints:
                    bottom = points[8][1] + 128
                else:
                    bottom = points[8][1]
                    bottom_ap = points[8][1] + 192
            elif 12 in joints:
                if 1 in joints or 2 in joints or 5 in joints:
                    top = points[8][1] - 192
                else:
                    top = points[8][1]
                    top_ap = points[8][1] - 192

                if 11 in joints or 14 in joints:
                    bottom = points[8][1] + 192
                elif 10 in joints or 13 in joints:
                    bottom = points[8][1] + 128
                else:
                    bottom = points[8][1]
                    bottom_ap = points[8][1] + 192
            else:
                cv2.imwrite(align_save_path3 + img_name, image)
                continue

            if top < 0:
                top_ap = abs(top)
                top = 0
            if bottom > 384:
                bottom_ap = bottom-384
                bottom = 384

            top = int(top)
            bottom = int(bottom)

            # if (6 in joints or 7 in joints or 13 in joints) and (3 in joints or 4 in joints or 10 in joints):
            if x_max-x_min < 0:
                cv2.imwrite(align_save_path3 + img_name, image)
                continue
            else:
                x_min = int(x_min - (x_max-x_min)/7)
                x_max = int(x_max + (x_max-x_min)/7)

                if x_min - (x_max-x_min)/7 < 0:
                    min_ap = abs(x_min)
                    x_min = 0

                if x_max + (x_max-x_min)/7 > 128:
                    max_ap = (x_max + (x_max-x_min)/7)-128
                    x_max = 128
            # else:
            #     cv2.imwrite(align_save_path3 + img_name, image)
            #     continue

            print('top: ', top,', bottom: ', bottom)
            print('min: ', x_min, ', max: ', x_max)

            crop_img = image[top:bottom, x_min:x_max]
            image = crop_img
            crop_img2 = image[:381, :]
            print(np.shape(crop_img2))


            top_ap = int(top_ap)
            bottom_ap = int(bottom_ap)
            min_ap = int(min_ap)
            max_ap = int(max_ap)
            if top_ap < 0:
                top_ap = 0

            print(top_ap, bottom_ap, min_ap, max_ap)

            if (top_ap + np.shape(image)[0] + bottom_ap)==0:
                cv2.imwrite(align_save_path3 + img_name, image)
                continue
            if (min_ap+np.shape(image)[1]+max_ap) == 0:
                cv2.imwrite(align_save_path3 + img_name, image)
                continue

            emptyImage = np.zeros((top_ap + np.shape(image)[0] + bottom_ap, min_ap+np.shape(image)[1]+max_ap, 3), np.uint8)
            print(np.shape(emptyImage))
            print(np.shape(image))
            # print(np.shape(emptyImage))
            # emptyImage[top_ap:top_ap + bottom-top, min_ap:min_ap+x_max-x_min] = image
            emptyImage[top_ap:top_ap + np.shape(image)[0], min_ap:min_ap+np.shape(image)[1]] = image

            print(np.shape(emptyImage[top_ap:top_ap + np.shape(image)[0], min_ap:min_ap+np.shape(image)[1]]))
            # emptyImage[:np.shape(image)[0], :np.shape(image)[1]] = image

            pic = cv2.resize(emptyImage, (128, 384), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(align_save_path1 + img_name, pic)
            # cv2.imwrite(align_save_path1 + img_name, crop_img)


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

if args.pose == True:
    img_path = args.pose_img_path + args.dataset + sub_folder
else:
    img_path = args.org_img_path + args.dataset + sub_folder

json_file = args.json_path + args.dataset + under_line + args.subset + '.json'

save_path = args.pose2_img_path + args.dataset + sub_folder

# draw_points(img_path, json_file, save_path)

align_save_path1 = args.edit_img_path + args.dataset + sub_folder + 'rearrange/'
align_save_path2 = args.edit_img_path + args.dataset + sub_folder + 'background/'
align_save_path3 = args.edit_img_path + args.dataset + sub_folder + 'occlusion/'
align_save_path4 = args.edit_img_path + args.dataset + sub_folder + 'nodetect/'

part_aligned(img_path, json_file, align_save_path1, align_save_path2, align_save_path3, align_save_path4)



