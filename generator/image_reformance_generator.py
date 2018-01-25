from skimage import io,transform
import numpy as np
from keras.preprocessing import image

#平移图片x，同时平移关键点y
def shift(x, y=None, w_limit=0., h_limit=0., row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    wshift = np.random.uniform(-1*w_limit, w_limit)
    hshift = np.random.uniform(-1*h_limit, h_limit)
    h, w = x.shape[row_axis], x.shape[col_axis] #读取图片的高和宽
    tx = hshift * h #高偏移大小，若不偏移可设为0，若向上偏移设为正数
    ty = wshift * w #宽偏移大小，若不偏移可设为0，若向左偏移设为正数
    translation_matrix = np.array([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]])
    transform_matrix = translation_matrix
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)

    if y is not None:
        transform_matrix[0:-1, -1] *= -1
        all_x = np.concatenate((y[0:5], y[10:15], y[20:25], y[30:35]))
        all_y = np.concatenate((y[5:10], y[15:20], y[25:30], y[35:40]))
        all_x = all_x.reshape((1, -1))
        all_y = all_y.reshape((1, -1))
        all_xy1 = np.concatenate((all_y, all_x, np.ones((1, all_x.shape[1]))), axis=0)
        y_1 = np.dot(transform_matrix, all_xy1)[0:-1, :]
        y[0:5] = y_1[1, 0:5]
        y[5:10] = y_1[0, 0:5]
        y[10:15] = y_1[1, 5:10]
        y[15:20] = y_1[0, 5:10]
        y[20:25] = y_1[1, 10:15]
        y[25:30] = y_1[0, 10:15]
        y[30:35] = y_1[1, 15:20]
        y[35:40] = y_1[0, 15:20]

    return x, y

#旋转图片x,同时旋转关键点y
def rotate(x, y=None, rotate_limit=0, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-1 * rotate_limit, rotate_limit)  # 逆时针旋转角度
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)

    if y is not None:
        all_x = np.concatenate((y[0:5], y[10:15], y[20:25], y[30:35]))
        all_y = np.concatenate((y[5:10], y[15:20], y[25:30], y[35:40]))
        all_x = all_x.reshape((1, -1))
        all_y = all_y.reshape((1, -1))
        all_xy1 = np.concatenate((all_x, all_y, np.ones((1, all_x.shape[1]))), axis=0)
        y_1 = np.dot(transform_matrix, all_xy1)[0:-1, :]
        y[0:5] = y_1[0, 0:5]
        y[5:10] = y_1[1, 0:5]
        y[10:15] = y_1[0, 5:10]
        y[15:20] = y_1[1, 5:10]
        y[20:25] = y_1[0, 10:15]
        y[25:30] = y_1[1, 10:15]
        y[30:35] = y_1[0, 15:20]
        y[35:40] = y_1[1, 15:20]

    return x, y


# 90°,180°,270°旋转图片x,同时旋转关键点y(!!!!!!y尺寸为1：1且范围为0:1!!!!!)
def rotate90n(x, y=None):
    theta = np.random.choice(np.arange(0, 4), 1) * 90
    x = transform.rotate(x, theta)
    if theta == 180:
        y = 1 - y
    elif theta == 90:
        y_t = y[0:5].copy()
        y[0:5] = y[5:10]
        y[5:10] = 1 - y_t

        y_t = y[10:15].copy()
        y[10:15] = y[15:20]
        y[15:20] = 1 - y_t

        y_t = y[20:25].copy()
        y[20:25] = y[25:30]
        y[25:30] = 1 - y_t

        y_t = y[30:35].copy()
        y[30:35] = y[35:40]
        y[35:40] = 1 - y_t
    elif theta == 270:
        y_t = y[0:5].copy()
        y[0:5] = 1 - y[5:10]
        y[5:10] = y_t

        y_t = y[10:15].copy()
        y[10:15] = 1 - y[15:20]
        y[15:20] = y_t

        y_t = y[20:25].copy()
        y[20:25] = 1 - y[25:30]
        y[25:30] = y_t

        y_t = y[30:35].copy()
        y[30:35] = 1 - y[35:40]
        y[35:40] = y_t

    return x, y

#带数据增强（平移，旋转）的生成器generator
def generate_reformance_kp(file_list, label_list, batch_size, shuffle=True, random_seed=None):
    while True:
        if shuffle:
            if random_seed!=None:
                random_seed+=1
                np.random.seed(random_seed)
            index=np.arange(file_list.shape[0])
            np.random.shuffle(index)
            file_list=file_list[index]
            label_list=label_list[index]
        count = 0
        x, y = [], []
        for i,path in enumerate(file_list):
            img=io.imread(path)
            y_temp=label_list[i,:]*224.
            x_temp = np.reshape(img, (224, 224, 1))
            x_temp, y_temp = rotate(x_temp, y_temp, 45)#旋转
            x_temp, y_temp = shift(x_temp, y_temp, 0.15, 0.15)#平移
            y_temp/=224.
            x_temp = x_temp.reshape((224,224))
            count += 1
            x.append(x_temp)
            y.append(y_temp)
            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, 224, 224, 1).astype("float32")
                y = np.array(y)
                yield x, y
                x, y = [], []