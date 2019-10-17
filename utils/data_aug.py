import numpy as np
import random

# Data_augmentation

def random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def erase_np(img, sl = 0.02, sh = 0.4, r1 = 0.3):

    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]
    area = width * height

    erase_area_low_bound = np.round(np.sqrt(sl * area * r1)).astype(np.int)
    erase_area_up_bound = np.round(np.sqrt((sh * area) / r1)).astype(np.int)
    if erase_area_up_bound < height:
        h_upper_bound = erase_area_up_bound
    else:
        h_upper_bound = height
    if erase_area_up_bound < width:
        w_upper_bound = erase_area_up_bound
    else:
        w_upper_bound = width

    h = np.random.randint(erase_area_low_bound, h_upper_bound)
    w = np.random.randint(erase_area_low_bound, w_upper_bound)

    # x1 = np.random.randint(0, height+1 - h)
    # y1 = np.random.randint(0, width+1 - w)

    x1 = np.random.randint(0, height - h)
    y1 = np.random.randint(0, width - w)
    img[x1:x1 + h, y1:y1 + w, :] = np.random.randint(0, 255, size=(h, w, channel)).astype(np.uint8)
    return img

def random_erase(batch, sl = 0.02, sh = 0.4, r1 = 0.3):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            half_h = batch[i].shape[0] // 2
            # w = batch[i].shape[1]
            if bool(random.getrandbits(1)):
                batch[i][:half_h, :, :] = erase_np(batch[i][:half_h, :, :], sl=sl, sh=sh, r1=r1)
            else:
                batch[i][half_h:, :, :] = erase_np(batch[i][half_h:, :, :], sl=sl, sh=sh, r1=r1)
    return batch

def data_augmentation(batch):
    batch = random_flip_leftright(batch)
    batch = random_crop(batch, [384, 128], 16)
    batch = random_erase(batch)
    return batch