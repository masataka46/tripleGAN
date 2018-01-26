import numpy as np
import os
from PIL import Image


def convert_to_10class(d):
    d_mod = np.zeros((len(d), 10), dtype=np.float32)
    for num, contents in enumerate(d):
        d_mod[num][int(contents)] = 1.0
    # debug
    # print("d_mod[100] =", d_mod[100])
    # print("d_mod[200] =", d_mod[200])

    return d_mod

def make_1_img(img_batch):  # for debug
    for num, ele in enumerate(img_batch):
        if num != 0:
            continue

        img_tmp = ele
        img_tmp = np.tile(img_tmp, (1, 1, 3)) * 255
        img_tmp = img_tmp.astype(np.uint8)
        image_PIL = Image.fromarray(img_tmp)
        image_PIL.save("./out_images_tripleGAN/debug_img_" + ".png")

    return
    
def make_output_img(image_array, sample_num_h, out_image_dir, epoch):

    # print("image_array.shape =", image_array.shape)
    # print("np.max(image_array) = ", np.max(image_array))
    # print("np.min(image_array) = ", np.min(image_array))
    # print("np.mean(image_array) = ", np.mean(image_array))

    wide_image = np.zeros((28 * sample_num_h, 28 * sample_num_h, 1), dtype=np.float32)
    for h in range(sample_num_h):
        for w in range(sample_num_h):
            for h_mnist in range(28):
                for w_mnist in range(28):
                    value_ = image_array[h * sample_num_h + w][h_mnist][w_mnist][0]
                    if value_ < 0:
                        wide_image[h * 28 + h_mnist][w * 28 + w_mnist][0] = 0.0
                    elif value_ > 1:
                        wide_image[h * 28 + h_mnist][w * 28 + w_mnist][0] = 1.0
                    else:
                        wide_image[h * 28 + h_mnist][w * 28 + w_mnist][0] = value_

    wide_image = np.tile(wide_image, (1, 1, 3)) * 255
    wide_image = wide_image.astype(np.uint8)
    wide_image_PIL = Image.fromarray(wide_image)
    wide_image_PIL.save(out_image_dir + "/resultImage_" + str(epoch) + ".png")

    small_image = (np.tile(image_array[0], (1, 1, 3)) * 255).astype(np.uint8)
    small_image_PIL = Image.fromarray(small_image)
    small_image_PIL.save(out_image_dir + "/resultImageSmall_" + str(epoch) + ".png")
    
    return




