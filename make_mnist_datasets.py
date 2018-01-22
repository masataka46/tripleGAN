import numpy as np
from PIL import Image
import utility as Utility

class Make_mnist_datasets():

    def __init__(self, filename_list, alpha_P):
        # def convert_to_10class(d):
        #     d_mod = np.zeros((len(d), 10), dtype=np.float32)
        #     for num, contents in enumerate(d):
        #         d_mod[num][int(contents)] = 1.0
        #     #debug
        #     print("d_mod[100] =", d_mod[100])
        #     print("d_mod[200] =", d_mod[200])
        #
        #     return d_mod

        # def make_1_img(img_batch):  # for debug
        #
        #     for num, ele in enumerate(img_batch):
        #         if num != 1234:
        #             continue
        #
        #         img_tmp = ele
        #         img_tmp = np.tile(img_tmp, (1, 1, 3)) * 255
        #         img_tmp = img_tmp.astype(np.uint8)
        #         image_PIL = Image.fromarray(img_tmp)
        #         image_PIL.save("./out_images_tripleGAN2/row_img_" + ".png")
        #
        #     return


        #load data from npz file...['test','test_labels','train','train_labels']
        train_img = np.load(filename_list[0])
        train_label = np.load(filename_list[1])
        test_img = np.load(filename_list[2])
        test_label = np.load(filename_list[3])
        
        print("train_img.shape =", train_img.shape)
        print("train_label.shape =", train_label.shape)
        print("test_img.shape =", test_img.shape)
        print("test_label.shape =", test_label.shape)

        
        # print('type(npz_xd["train"])=', type(npz_xd["train"]))
        # print('type(npz_xd["test"])=', type(npz_xd["test"]))
        # print('type(npz_xd["train_labels"])=', type(npz_xd["train_labels"]))
        # print('type(npz_xd["test_labels"])=', type(npz_xd["test_labels"]))
        # print('type(npz_xd["train.shape"])=', npz_xd["train"].shape)
        # print('type(npz_xd["test.shape"])=', npz_xd["test"].shape)
        # print('type(npz_xd["train_labels.shape"])=', npz_xd["train_labels"].shape)
        # print('type(npz_xd["test_labels.shape"])=', npz_xd["test_labels"].shape)
        #make input data, target data
        x_train = train_img.reshape(60000, 28, 28, 1).astype(np.float32)
        l_train = train_label.reshape(60000, 1)
        x_test = test_img.reshape(10000, 28, 28, 1).astype(np.float32)
        l_test = test_label.reshape(10000, 1)
        #
        d_train = Utility.convert_to_10class(l_train)
        d_test = Utility.convert_to_10class(l_test)

        # d_train = convert_to_10class(l_train)
        # d_test = convert_to_10class(l_test)

        #debug
        print("x_train.shape = ", x_train.shape)
        print("d_train.shape = ", d_train.shape)
        print("x_test.shape = ", x_test.shape)
        print("d_test.shape = ", d_test.shape)



        # x_train_mod = np.zeros(x_train.shape, dtype=np.float32)
        # d_train_mod = np.zeros(d_train.shape, dtype=np.float32)
        # x_test_mod = np.zeros(x_test.shape, dtype=np.float32)
        # d_test_mod = np.zeros(d_test.shape, dtype=np.float32)
        #change (0,0,...,0,1,1,...,1,.....,9,9,...,9) to (0,1,2,..,9,0,1,2,..,9,...,0,1,..,9)
        # for num, row in enumerate(x_train):
        #     x_train_mod[(num % 6000) * 10 + (num // 6000)] = row
        #     d_train_mod[(num % 6000) * 10 + (num // 6000)] = d_train[num]
        # 
        # for num, row in enumerate(x_test):
        #     x_test_mod[(num % 1000) * 10 + (num // 1000)] = row
        #     d_test_mod[(num % 1000) * 10 + (num // 1000)] = d_test[num]
            
        #debug
        print("x_train.shape = ", x_train.shape)
        print("d_train.shape = ", d_train.shape)
        print("x_test.shape = ", x_test.shape)
        print("d_test.shape = ", d_test.shape)
        print("d_train[0] = ", d_train[0])
        print("d_train[1] = ", d_train[1])
        print("d_train[2] = ", d_train[2])
        print("d_train[3] = ", d_train[3])
        print("d_train[4] = ", d_train[4])
        print("d_train[5] = ", d_train[5])
        print("d_train[6] = ", d_train[6])
        print("d_train[7] = ", d_train[7])
        print("d_train[8] = ", d_train[8])
        print("d_train[9] = ", d_train[9])
        print("d_train[10] = ", d_train[10])
        print("d_train[6999] = ", d_train[6999])
        print("d_train[1234] = ", d_train[1234])
        print("d_test[0] = ", d_test[0])
        print("d_test[1] = ", d_test[1])
        print("d_test[2] = ", d_test[2])
        print("d_test[9] = ", d_test[9])
        print("d_test[10] = ", d_test[10])
        print("d_test[1234] = ", d_test[1234])

        # make_1_img(x_train)
        Utility.make_1_img(x_train)
        
        self.real_num = int(len(x_train) / (1 + alpha_P)) #40,000
        self.else_num = len(x_train) - self.real_num #20,000

        self.img_real = x_train[0:self.real_num]
        self.img_cla = x_train[self.real_num :]
        # img_gen = img[int(len(img) / 2) + int(len(img) / 2 * alpha_P) :]

        self.label_real = d_train[0:self.real_num]
        # label_gen = label[int(len(img) / 2) + int(len(img) / 2 * alpha_P) :]
        print("self.img_real.shape = ", self.img_real.shape)
        print("self.img_cla.shape = ", self.img_cla.shape)
        print("self.label_real.shape = ", self.label_real.shape)


    def make_data_for_1_epoch(self):

        randInt_real = np.random.permutation(self.real_num)
        randInt_else = np.random.permutation(self.else_num)
        self.img_real_1epoch = self.img_real[randInt_real]
        self.img_cla_1epoch = self.img_cla[randInt_else]
        self.label_real_1epoch = self.label_real[randInt_real]
        # print("img_real_tmp.shape =(state 1) ", img_real_tmp.shape)
        # return img_real_tmp, img_cla_tmp, label_real_tmp
        return len(self.img_real_1epoch)


    def get_data_for_1_batch(self, i, batchsize, alpha_P):
        img_real_batch = self.img_real_1epoch[i:i + batchsize]
        img_cla_batch = self.img_cla_1epoch[int(i * alpha_P) : int(i * alpha_P) + int(batchsize * alpha_P)]
        label_real_batch = self.label_real_1epoch[i:i + batchsize]

        return img_real_batch, img_cla_batch, label_real_batch


    def convert_to_10class_(self, d):
        d_mod = np.zeros((len(d), 10), dtype=np.float32)
        for num, contents in enumerate(d):
            d_mod[num][int(contents)] = 1.0
        # debug
        # print("d_mod[2] =", d_mod[2])
        # print("d_mod[4] =", d_mod[4])
        return d_mod


    def print_img_and_label(self, img_batch, label_batch, int0to9):# for debug

        for num, ele in enumerate(img_batch):
            if num % 10 != int0to9:
                continue

            print("label_batch[", num, "]=", label_batch[num])

            label_num = 0
            for num2, ele2 in enumerate(label_batch[num]):
                if int(ele2) == 1:
                    label_num = num2

            img_tmp = ele
            img_tmp = np.tile(img_tmp, (1, 1, 3)) * 255
            img_tmp = img_tmp.astype(np.uint8)
            image_PIL = Image.fromarray(img_tmp)
            image_PIL.save("./out_images_Debug/debug_img_" + str(num) + "_" + str(label_num) + ".png")
            # image_PIL.save("./out_images_Debug/debug_img_" + str(label_num) + "_.png")

        return

