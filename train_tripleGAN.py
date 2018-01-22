import numpy as np
import os
import tensorflow as tf
from PIL import Image
import utility as Utility

from make_mnist_datasets import Make_mnist_datasets

#global variants
batchsize = 100
data_size = 6000
noise_num = 100
class_num = 10
n_epoch = 1000
l2_norm_lambda = 0.001
alpha_P = 0.5
alpha_pseudo = 0.1
alpha_apply_thr = 200

keep_prob_rate = 0.5

mnist_file_name = ["mnist_train_img.npy", "mnist_train_label.npy", "mnist_test_img.npy", "mnist_test_label.npy"]
board_dir_name = "data20" #directory for tensorboard
seed = 1234
np.random.seed(seed=seed)

# adam_b1_d = 0.5
# adam_b1_c = 0.5
# adam_b1_g = 0.5

make_mnist = Make_mnist_datasets(mnist_file_name, alpha_P)

out_image_dir = './out_images_tripleGAN2' #output image file
out_model_dir = './out_models_tripleGAN' #output model file
try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
    os.mkdir('./out_images_Debug') #for debug
except:
    pass


def leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def gaussian_noise(input, std): #used at discriminator
    noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32, seed=seed)
    return input + noise

#generator------------------------------------------------------------------
# wg1 = tf.Variable(tf.random_normal([class_num + noise_num, 500], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wg1')
# bg1 = tf.Variable(tf.zeros([500]), name='bg1')
# scaleg2 = tf.Variable(tf.ones([500]), name='sg2')
# betag2 = tf.Variable(tf.zeros([500]), name='beg2')
# wg3 = tf.Variable(tf.random_normal([500, 500], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wg3')
# bg3 = tf.Variable(tf.zeros([500]), name='bg3')
# scaleg4 = tf.Variable(tf.ones([500]), name='sg4')
# betag4 = tf.Variable(tf.zeros([500]), name='beg4')
# wg5 = tf.Variable(tf.random_normal([500, 784], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wg5')
# bg5 = tf.Variable(tf.zeros([784]), name='bg5')

def generator(y, z):
    with tf.variable_scope('g'):
        wg1 = tf.Variable(tf.random_normal([class_num + noise_num, 500], mean=0.0, stddev=0.05, seed=seed),
                          dtype=tf.float32, name='wg1')
        bg1 = tf.Variable(tf.zeros([500]), name='bg1')
        scaleg2 = tf.Variable(tf.ones([500]), name='sg2')
        betag2 = tf.Variable(tf.zeros([500]), name='beg2')
        wg3 = tf.Variable(tf.random_normal([500, 500], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wg3')
        bg3 = tf.Variable(tf.zeros([500]), name='bg3')
        scaleg4 = tf.Variable(tf.ones([500]), name='sg4')
        betag4 = tf.Variable(tf.zeros([500]), name='beg4')
        wg5 = tf.Variable(tf.random_normal([500, 784], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wg5')
        bg5 = tf.Variable(tf.zeros([784]), name='bg5')
        #concat label and noise
        concat0 = tf.concat([y, z], axis=1, name='G_concat0')

        #layer1 linear
        fc1 = tf.matmul(concat0, wg1, name='G_matmul1') + bg1
        #softplus function
        sp1 = tf.log(tf.clip_by_value(1 + tf.exp(fc1), 1e-10, 1.0), name='G_softmax1')

        #layer2 batch normalization
        batch_mean2, batch_var2 = tf.nn.moments(sp1, [0])
        bn2 = tf.nn.batch_normalization(sp1, batch_mean2, batch_var2, betag2, scaleg2 , 0.0001, name='G_BN2')

        #layer3 linear
        fc3 = tf.matmul(bn2, wg3, name='G_matmul3') + bg3
        #softplus function
        sp3 = tf.log(tf.clip_by_value(1 + tf.exp(fc3), 1e-10, 1.0), name='G_softmax3')

        #layer4 batch normalization
        batch_mean4, batch_var4 = tf.nn.moments(sp3, [0])
        bn4 = tf.nn.batch_normalization(sp3, batch_mean4, batch_var4, betag4, scaleg4 , 0.0001, name='G_BN4')

        #layer5 linear
        fc5 = tf.matmul(bn4, wg5, name='G_matmul5') + bg5
        #sigmoid function
        sig5 = tf.nn.sigmoid(fc5, name='G_sigmoid5')

        #reshape to 28x28 image
        x_gen = tf.reshape(sig5, [-1, 28, 28, 1])

        return x_gen, y


#discriminator-----------------------------------------------------------------
wd1 = tf.Variable(tf.random_normal([794, 1000], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wd1')
bd1 = tf.Variable(tf.zeros([1000]), name='bd1')
wd2 = tf.Variable(tf.random_normal([1000, 500], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wd2')
bd2 = tf.Variable(tf.zeros([500]), name='bd2')
wd3 = tf.Variable(tf.random_normal([500, 250], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wd3')
bd3 = tf.Variable(tf.zeros([250]), name='bd3')
wd4 = tf.Variable(tf.random_normal([250, 250], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wd4')
bd4 = tf.Variable(tf.zeros([250]), name='bd4')
wd5 = tf.Variable(tf.random_normal([250, 250], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wd5')
bd5 = tf.Variable(tf.zeros([250]), name='bd5')
wd6 = tf.Variable(tf.random_normal([250, 1], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wd6')
bd6 = tf.Variable(tf.zeros([1]), name='bd6')

def discriminator(x, y):
    with tf.variable_scope('discriminator'):
        x_reshape = tf.reshape(x, [-1, 28 * 28])
        # concat image and label
        concat0 = tf.concat([x_reshape, y], axis=1, name='D_concat0')

        # layer1 linear
        #gaussian noise
        gn1 = gaussian_noise(concat0, 0.3)
        #fully-connected
        fc1 = tf.matmul(gn1, wd1, name='D_matmul1') + bd1
        # leakyReLU function
        lr1 = leaky_relu(fc1, alpha=0.2)

        # layer2 linear
        #gaussian noise
        gn2 = gaussian_noise(lr1, 0.5)
        #fully-connected
        fc2 = tf.matmul(gn2, wd2, name='D_matmul2') + bd2
        # leakyReLU function
        lr2 = leaky_relu(fc2, alpha=0.2)

        # layer3 linear
        #gaussian noise
        gn3 = gaussian_noise(lr2, 0.5)
        #fully-connected
        fc3 = tf.matmul(gn3, wd3, name='D_matmul3') + bd3
        # leakyReLU function
        lr3 = leaky_relu(fc3, alpha=0.2)

        # layer4 linear
        #gaussian noise
        gn4 = gaussian_noise(lr3, 0.5)
        #fully-connected
        fc4 = tf.matmul(gn4, wd4, name='D_matmul4') + bd4
        # leakyReLU function
        lr4 = leaky_relu(fc4, alpha=0.2)

        # layer5 linear
        #gaussian noise
        gn5 = gaussian_noise(lr4, 0.5)
        #fully-connected
        fc5 = tf.matmul(gn5, wd5, name='D_matmul5') + bd5
        # leakyReLU function
        lr5 = leaky_relu(fc5, alpha=0.2)

        # layer6 linear
        #gaussian noise
        gn6 = gaussian_noise(lr5, 0.5)
        #fully-connected
        fc6 = tf.matmul(gn6, wd6, name='D_matmul6') + bd6
        # softplus function
        out_dis = tf.nn.sigmoid(fc6, name='D_sigmoid')

        return out_dis


#classifier-----------------------------------------------------------------
wc1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wc1')
bc1 = tf.Variable(tf.zeros([32]), name='bc1')
wc2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wc2')
bc2 = tf.Variable(tf.zeros([64]), name='bc2')
wc3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wc3')
bc3 = tf.Variable(tf.zeros([64]), name='bc3')
wc4 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wc4')
bc4 = tf.Variable(tf.zeros([128]), name='bc4')
wc5 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wc5')
bc5 = tf.Variable(tf.zeros([128]), name='bc5')
# wc6 = tf.Variable(tf.truncated_normal([1, 1, 128, class_num], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32)
# bc6 = tf.Variable(tf.zeros([class_num]))
wc6 = tf.Variable(tf.random_normal([128, 10], mean=0.0, stddev=0.05, seed=seed), dtype=tf.float32, name='wc6')
bc6 = tf.Variable(tf.zeros([10]), name='bc6')

def classifier(xc, keep_prob):
    with tf.variable_scope('classifier'):
        #layer1 convolution
        conv1 = tf.nn.conv2d(xc, wc1, strides=[1, 1, 1, 1], padding="SAME", name='C_conv1') + bc1
        # relu function
        conv1_relu = tf.nn.relu(conv1)
        #max pooling
        conv1_pool = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        #drop out
        conv1_drop = tf.nn.dropout(conv1_pool, keep_prob)

        #layer2 convolution
        conv2 = tf.nn.conv2d(conv1_drop, wc2, strides=[1, 1, 1, 1], padding="SAME", name='C_conv2') + bc2
        # relu function
        conv2_relu = tf.nn.relu(conv2)

        #layer3 convolution
        conv3 = tf.nn.conv2d(conv2_relu, wc3, strides=[1, 1, 1, 1], padding="SAME", name='C_conv3') + bc3
        # relu function
        conv3_relu = tf.nn.relu(conv3)
        #max pooling
        conv3_pool = tf.nn.max_pool(conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        #drop out
        conv3_drop = tf.nn.dropout(conv3_pool, keep_prob)

        #layer4 convolution
        conv4 = tf.nn.conv2d(conv3_drop, wc4, strides=[1, 1, 1, 1], padding="SAME", name='C_conv4') + bc4
        # relu function
        conv4_relu = tf.nn.relu(conv4)

        #layer5 convolution
        conv5 = tf.nn.conv2d(conv4_relu, wc5, strides=[1, 1, 1, 1], padding="SAME", name='C_conv5') + bc5
        # relu function
        conv5_relu = tf.nn.relu(conv5)

        # conv6 = tf.nn.conv2d(conv5_relu, wc6, strides=[1, 1, 1, 1], padding="SAME") + bc6
        # global average pooling.... reduce mean
        ap5 = tf.reduce_mean(conv5_relu, axis=[1, 2], name='C_global_average')

        #layer6 full-connected
        fc6 = tf.matmul(ap5, wc6, name='C_matmul6') + bc6

        #softmax
        yc = tf.nn.softmax(fc6, name='C_softmax')

        # tf.summary.histogram("Cconv1", conv1)
        # tf.summary.histogram("Cconv2", conv2)
        # tf.summary.histogram("Cconv3", conv3)
        # tf.summary.histogram("Cconv4", conv4)
        # tf.summary.histogram("Cconv5", conv5)
        # tf.summary.histogram("Cap5", ap5)
        tf.summary.histogram("Cfc6", fc6)
        tf.summary.histogram("yc", yc)

        return xc, yc

# placeholder
yg_ = tf.placeholder(tf.float32, [None, class_num], name='yg_') #label to generator
z_ = tf.placeholder(tf.float32, [None, noise_num], name='z_') #noise to generator
xc1_ = tf.placeholder(tf.float32, [None, 28, 28, 1], name='xc1_') #labeled image to classifier
xc2_ = tf.placeholder(tf.float32, [None, 28, 28, 1], name='xc2_') #unlabeled image to classifier
yd_ = tf.placeholder(tf.float32, [None, class_num], name='yd_') #label to discriminator
xd_ = tf.placeholder(tf.float32, [None, 28, 28, 1], name='xd_') #labeled image to discriminator
d_dis_g_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_g_') #target of discriminator related to generator
d_dis_r_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_r_') #target of discriminator related to real image
d_dis_c_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_c_') #target of discriminator related to classifier
yc1_ = tf.placeholder(tf.float32, [None, class_num], name='yc1_') #target label of classifier related to real image
alpha_p_flag_ = tf.placeholder(tf.float32, name='alpha_p_flag_') #(0,1) apply alpha pseudo or not

keep_prob_ = tf.placeholder(tf.float32, name='keep_prob_') #dropout rate

# stream around generator
x_gen, y_gen = generator(yg_, z_)

# stream around classifier
x_cla_0, y_cla_0 = classifier(x_gen, keep_prob_) # from generator
x_cla_1, y_cla_1 = classifier(xc1_, keep_prob_) # real image labeled
x_cla_2, y_cla_2 = classifier(xc2_, keep_prob_) # real image unlabeled

# loss_RP = - tf.reduce_mean(y_gen * tf.log(y_cla_0)) #loss in case generated image
# loss_RL = - tf.reduce_mean(yc1_ * tf.log(y_cla_1)) #loss in case real image
loss_RP = - tf.reduce_mean(y_gen * tf.log(tf.clip_by_value(y_cla_0, 1e-10, 1.0)), name='Loss_RP') #loss in case generated image
loss_RL = - tf.reduce_mean(yc1_ * tf.log(tf.clip_by_value(y_cla_1, 1e-10, 1.0)), name='Loss_RL') #loss in case real image

#stream around discriminator
out_dis_g = discriminator(x_gen, y_gen) #from generator
out_dis_r = discriminator(xd_, yd_) #real image and label
out_dis_c = discriminator(x_cla_2, y_cla_2) #from classifier

loss_dis_g = tf.reduce_mean(tf.square(out_dis_g - d_dis_g_), name='Loss_dis_gen') #loss related to generator
loss_dis_r = tf.reduce_mean(tf.square(out_dis_r - d_dis_r_), name='Loss_dis_rea') #loss related to real imaeg
loss_dis_c = tf.reduce_mean(tf.square(out_dis_c - d_dis_c_), name='Loss_dis_cla') #loss related to classifier

norm_L2 = tf.nn.l2_loss(wd1) + tf.nn.l2_loss(wd2) + tf.nn.l2_loss(wd3) + tf.nn.l2_loss(wd4) + tf.nn.l2_loss(wd5)\
    + tf.nn.l2_loss(wd6)


#total loss of discriminator
loss_dis_total = loss_dis_r + alpha_P * loss_dis_c + (1 - alpha_P) * loss_dis_g + l2_norm_lambda * norm_L2

#total loss of classifier
loss_cla_total = alpha_P * loss_dis_c + loss_RL + alpha_p_flag_ * alpha_pseudo * loss_RP

#total loss of generator
loss_gen_total = (1 - alpha_P) * loss_dis_g

# tf.summary.scalar('loss_dis_total', loss_dis_total)
tf.summary.histogram("wc1", wc1)
# tf.summary.histogram("wc2", wc2)
# tf.summary.histogram("wc3", wc3)
# tf.summary.histogram("wc4", wc4)
# tf.summary.histogram("wc5", wc5)
# tf.summary.histogram("wc6", wc6)
# tf.summary.histogram("bc1", bc1)
# tf.summary.histogram("bc2", bc2)
# tf.summary.histogram("bc3", bc3)
# tf.summary.histogram("bc4", bc4)
# tf.summary.histogram("bc5", bc5)
tf.summary.histogram("bc6", bc6)



tf.summary.scalar('loss_cla_total', loss_cla_total)

# tf.summary.scalar('loss_gen_total', loss_gen_total)
merged = tf.summary.merge_all()

g_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')

train_dis = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(loss_dis_total,
                                    var_list=[wd1, wd2, wd3, wd4, wd5, wd6, bd1, bd2, bd3, bd4, bd5, bd6]
                                                                            , name='Adam_dis')
train_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(loss_gen_total,
                                    # var_list=[wg1, wg3, wg5, bg1, bg3, bg5, betag2, scaleg2, betag4, scaleg4]
                                    var_list=g_vars
                                                                            , name='Adam_gen')
train_cla = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(loss_cla_total,
                                    var_list=[wc1, wc2, wc3, wc4, wc5, wc6, bc1, bc2, bc3, bc4, bc5, bc6]
                                                                            , name='Adam_cla')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(board_dir_name, sess.graph)

#training loop
for epoch in range(0, n_epoch):
    sum_loss_dis = np.float32(0)
    sum_loss_gen = np.float32(0)
    sum_loss_cla = np.float32(0)
    sum_accu_cla = np.float32(0)

    len_img_real = make_mnist.make_data_for_1_epoch()
    #debug
    print("make_mnist.img_real_1epoch.shape = ", make_mnist.img_real_1epoch.shape)
    print("make_mnist.img_cla_1epoch.shape = ", make_mnist.img_cla_1epoch.shape)

    for i in range(0, len_img_real, batchsize):
        img_real_batch, img_cla_batch, label_real_batch = make_mnist.get_data_for_1_batch(i, batchsize, alpha_P)

        #debug
        if epoch == 0 and i == 0:
            make_mnist.print_img_and_label(img_real_batch, label_real_batch, 7)

        #cal each batchsize
        len_real_batch = len(img_real_batch)
        len_cla_batch = len(img_cla_batch)
        len_gen_batch = int(len(img_real_batch) * alpha_P)

        z = np.random.uniform(0, 1, len_gen_batch * noise_num)
        z = z.reshape(-1, noise_num).astype(np.float32)

        label_gen_int = np.random.randint(0, class_num, len_gen_batch)
        label_gen = make_mnist.convert_to_10class_(label_gen_int)


        d_dis_g_1_ = np.array([1.0], dtype=np.float32).reshape(1, 1)
        d_dis_g_1 = np.tile(d_dis_g_1_, (len_gen_batch, 1))

        d_dis_g_0_ = np.array([0.0], dtype=np.float32).reshape(1, 1)
        d_dis_g_0 = np.tile(d_dis_g_0_, (len_gen_batch, 1))

        d_dis_r_1 = np.array([1.0], dtype=np.float32).reshape(1, 1)
        d_dis_r = np.tile(d_dis_r_1, (len_real_batch, 1))

        d_dis_c_1_ = np.array([1.0], dtype=np.float32).reshape(1, 1)
        d_dis_c_1 = np.tile(d_dis_c_1_, (len_cla_batch, 1))

        d_dis_c_0_ = np.array([0.0], dtype=np.float32).reshape(1, 1)
        d_dis_c_0 = np.tile(d_dis_c_0_, (len_cla_batch, 1))

        #debug
        # print("img_real.shape = ", img_real_batch.shape)
        # print("img_cla_batch.shape = ", img_cla_batch.shape)
        # print("img_real_batch.shape = ", img_real_batch.shape)
        # # print("img_cla_batch.shape = ", img_cla_batch.shape)
        # #debug
        # # out_dis_c_debug = sess.run(out_dis_c, feed_dict={z_:z, yg_:label_gen, yd_: label_real_batch,
        # #                                                  xd_: img_real_batch,
        # #                                xc2_: img_cla_batch, d_dis_g_: d_dis_g_0, d_dis_r_: d_dis_r_1,
        # #                                d_dis_c_:d_dis_c_0, keep_prob_:0.8})
        # # print("out_dis_c_debug.shape = ", out_dis_c_debug.shape)
        # print("d_dis_c_0.shape = ", d_dis_c_0.shape)

        sess.run(train_gen, feed_dict={z_:z, yg_:label_gen, d_dis_g_: d_dis_g_1})
        sess.run(train_dis, feed_dict={z_:z, yg_:label_gen, yd_: label_real_batch, xd_: img_real_batch,
                                       xc2_: img_cla_batch, d_dis_g_: d_dis_g_0, d_dis_r_: d_dis_r_1,
                                       d_dis_c_:d_dis_c_0, keep_prob_:keep_prob_rate})

        if epoch > alpha_apply_thr:
            sess.run(train_cla, feed_dict={z_:z, yg_:label_gen, xc1_: img_real_batch, xc2_: img_cla_batch,
                                       yc1_: label_real_batch, d_dis_c_: d_dis_c_1,keep_prob_:keep_prob_rate,
                                       alpha_p_flag_:1.0})
        else:
            sess.run(train_cla, feed_dict={z_: z, yg_: label_gen, xc1_: img_real_batch, xc2_: img_cla_batch,
                                           yc1_: label_real_batch, d_dis_c_: d_dis_c_1, keep_prob_: keep_prob_rate,
                                           alpha_p_flag_: 0.0})

        loss_gen_total_ = sess.run(loss_gen_total, feed_dict={z_:z, yg_:label_gen, d_dis_g_: d_dis_g_1})

        loss_dis_total_ = sess.run(loss_dis_total, feed_dict={z_:z, yg_:label_gen, yd_: label_real_batch,
                                        xd_: img_real_batch, xc2_: img_cla_batch, d_dis_g_: d_dis_g_0,
                                        d_dis_r_: d_dis_r_1, d_dis_c_:d_dis_c_0, keep_prob_:1.0})

        loss_cla_total_ = sess.run(loss_cla_total, feed_dict={z_:z, yg_:label_gen, xc1_: img_real_batch,
                                        xc2_: img_cla_batch,yc1_: label_real_batch,
                                                              d_dis_c_: d_dis_c_1, keep_prob_:1.0, alpha_p_flag_: 0.0})

        #debug
        merged_ = sess.run(merged, feed_dict={z_:z, yg_:label_gen, xc1_: img_real_batch,
                                        xc2_: img_cla_batch,yc1_: label_real_batch,
                                                              d_dis_c_: d_dis_c_1, keep_prob_:1.0, alpha_p_flag_: 0.0})


        summary_writer.add_summary(merged_, epoch)

        sum_loss_gen += loss_gen_total_
        sum_loss_dis += loss_dis_total_
        sum_loss_cla += loss_cla_total_
    print("epoch =", epoch , ", sum_loss_gen =", sum_loss_gen, ", sum_loss_dis =", sum_loss_dis,
          ", sum_loss_cla =", sum_loss_cla)

    if epoch % 10 == 0:

        sample_num_h = 10
        sample_num = sample_num_h ** 2

        # z_test = np.random.uniform(0, 1, sample_num * noise_num)
        z_test = np.random.uniform(0, 1, sample_num_h * noise_num).reshape(sample_num_h, 1, noise_num)
        z_test = np.tile(z_test, (1, sample_num_h, 1))
        z_test = z_test.reshape(-1, sample_num).astype(np.float32)
        # label_gen_int = np.random.randint(0, class_num, sample_num)
        label_gen_int = np.arange(10).reshape(10, 1).astype(np.float32)
        label_gen_int = np.tile(label_gen_int, (1, 10)).reshape(sample_num)

        # print("label_gen_int =", label_gen_int)
        label_gen_test = make_mnist.convert_to_10class_(label_gen_int)
        # print("label_gen_test =", label_gen_test)
        gen_images = sess.run(x_gen, feed_dict={z_:z_test, yg_:label_gen_test})
        # print("gen_images.shape =", gen_images.shape)
        # print("np.max(gen_images) = ", np.max(gen_images))
        # print("np.min(gen_images) = ", np.min(gen_images))
        # print("np.mean(gen_images) = ", np.mean(gen_images))

        Utility.make_output_img(gen_images, sample_num_h, out_image_dir, epoch)
        #debug
        # if epoch == 20:
        #     wide_image = np.tile(gen_images, (1, 1, 1, 3)) * 255
        #     wide_image = wide_image.astype(np.uint8)
        #     for i in range(100):
        #         wide_image_PIL = Image.fromarray(wide_image[i])
        #         wide_image_PIL.save(out_image_dir + "/resultImage_ind_" + str(i) + ".png")
        #
        # wide_image = np.zeros((28 * sample_num_h, 28 * sample_num_h, 1), dtype=np.float32)
        # for h in range(sample_num_h):
        #     for w in range(sample_num_h):
        #         for h_mnist in range(28):
        #             for w_mnist in range(28):
        #                 value_ = gen_images[h * sample_num_h + w][h_mnist][w_mnist][0]
        #                 if value_ < 0:
        #                     wide_image[h * 28 + h_mnist][w * 28 + w_mnist][0] = 0.0
        #                 elif value_ > 1:
        #                     wide_image[h * 28 + h_mnist][w * 28 + w_mnist][0] = 1.0
        #                 else:
        #                     wide_image[h * 28 + h_mnist][w * 28 + w_mnist][0] = value_
        #
        # wide_image = np.tile(wide_image, (1, 1, 3)) * 255
        # wide_image = wide_image.astype(np.uint8)
        # wide_image_PIL = Image.fromarray(wide_image)
        # wide_image_PIL.save(out_image_dir + "/resultImage_" + str(epoch) + ".png")
        #
        # small_image = (np.tile(gen_images[0], (1, 1, 3)) * 255).astype(np.uint8)
        # small_image_PIL = Image.fromarray(small_image)
        # small_image_PIL.save(out_image_dir + "/resultImageSmall_" + str(epoch) + ".png")





