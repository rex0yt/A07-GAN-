import os, time, random, itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2


class ConvoGan:
    def __init__(self):
        self.name = 'GAN_MINIST'  # 保存图片的命名
        self.save_img = './FakeImg'  # 保存图片的文件夹
        self.save_dir = './model'  # 保存模型的文件夹
        self.checkpoint_name = 'train.ckpt'  # 保存模型的命名

        # 如果不存在路径则创建
        if not os.path.isdir(self.save_img):
            os.mkdir(self.save_img)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        # 初始化
        self.IMAGE_SIZE = 28
        self.onehot = np.eye(10)

        self.batch_size = 100   # 批大小
        self.step = 30  # 一共迭代次数
        self.global_step = tf.Variable(0, trainable=False)  # 设置一个全局的计数器
        self.lr = tf.train.exponential_decay(0.0001, self.global_step, 500, 0.95, staircase=True)   # 设置学习率
        # 加载数据集Batch大小：100
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

        self.sess = tf.InteractiveSession()

    def leaky_relu(self, X, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * X + f2 * tf.abs(X)

    # 定义生成器网络
    def Generator(self, x, labels, Training=True, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse):
            # 初始化参数
            W = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b = tf.constant_initializer(0.0)
            # 把数据和标签进行连接
            concat = tf.concat([x, labels], 3)
            # 第一次反卷积,卷积核大小为7*7，输出维度256
            out_1 = tf.layers.conv2d_transpose(concat, 256, [7, 7], strides=(1, 1), padding='valid',
                                               kernel_initializer=W,
                                               bias_initializer=b)
            out_1 = tf.layers.batch_normalization(out_1, training=Training)  # batch norm
            out_1 = self.leaky_relu(out_1, 0.2)
            # 第二次反卷积，卷积核大小为5*5，输出维度128
            out_2 = tf.layers.conv2d_transpose(out_1, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=W,
                                               bias_initializer=b)
            out_2 = tf.layers.batch_normalization(out_2, training=Training)  # batch norm
            out_2 = self.leaky_relu(out_2, 0.2)
            # 第三次反卷积，卷积核大小5*5，输出维度1
            out_3 = tf.layers.conv2d_transpose(out_2, 1, [5, 5], strides=(2, 2), padding='same', kernel_initializer=W,
                                               bias_initializer=b)
            out_3 = tf.nn.tanh(out_3)
            return out_3

    # 定义判别器网络结构
    def Discriminator(self, x, real, Training=True, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            # 初始化参数
            W = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b = tf.constant_initializer(0.0)
            # 把数据和标签进行连接
            concat = tf.concat([x, real], 3)
            # 第一次卷积 卷积核为5*5 输出维度为128
            out_1 = tf.layers.conv2d(concat, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=W,
                                     bias_initializer=b)
            out_1 = self.leaky_relu(out_1, 0.2)
            # 第二次卷积 卷积核为5*5 输出维度256
            out_2 = tf.layers.conv2d(out_1, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=W,
                                     bias_initializer=b)
            out_2 = tf.layers.batch_normalization(out_2, training=Training)  # batch norm
            out_2 = self.leaky_relu(out_2, 0.2)
            # 第三次卷积，卷积和为7*7，输出维度为1
            out_3 = tf.layers.conv2d(out_2, 1, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=W)
            logits = tf.nn.sigmoid(out_3)
            return logits, out_3

    # 生成特定数字的图像
    def showi(self, num, path='aa'):
        sess = self.sess
        onehot = self.onehot
        IMAGE_SIZE = self.IMAGE_SIZE

        noise_ = np.random.normal(0, 1, (1, 1, 1, 100))
        fixed_noise_ = noise_

        fixed_label_ = np.zeros((10, 1))
        fixed_label_[num] = 1
        fixed_label_ = np.reshape(fixed_label_, (1, 1, 1, 10))

        test_image = sess.run(self.G_noise, {self.noise: fixed_noise_, self.labels: fixed_label_, self.Training: False})
        img = np.reshape(test_image, (IMAGE_SIZE, IMAGE_SIZE))

        # cv2.imwrite(path, ((img+1)*255))     # 文件名 变量 (-1,1)to(0,255)

        plt.imshow(img, cmap='gray')
        plt.savefig(path)
        return (img+1)*255


    # 训练网络时生成并保存图片
    def show_result(self, num_epoch, show=False, save=False, path=None):
        sess = self.sess
        onehot = self.onehot
        IMAGE_SIZE = self.IMAGE_SIZE

        noise_ = np.random.normal(0, 1, (10, 1, 1, 100))
        fixed_noise_ = noise_
        fixed_label_ = np.zeros((10, 1))

        # 用于最后显示十组图像
        for i in range(9):
            fixed_noise_ = np.concatenate([fixed_noise_, noise_], 0)
            temp = np.ones((10, 1)) + i
            fixed_label_ = np.concatenate([fixed_label_, temp], 0)
        fixed_label_ = onehot[fixed_label_.astype(np.int32)].reshape((100, 1, 1, 10))

        test_images = sess.run(self.G_noise, {self.noise: fixed_noise_, self.labels: fixed_label_, self.Training: False})
        size_figure_grid = 10
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        for k in range(10 * 10):
            i = k // 10
            j = k % 10
            ax[i, j].cla()
            ax[i, j].imshow(np.reshape(test_images[k], (IMAGE_SIZE, IMAGE_SIZE)), cmap='gray')
        label = 'Step {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')
        if save:
            plt.savefig(path)
        if show:
            plt.show()
        else:
            plt.close()

    def createModel(self):
        IMAGE_SIZE = self.IMAGE_SIZE
        batch_size = self.batch_size

        self.x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1))
        self.noise = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
        self.labels = tf.placeholder(tf.float32, shape=(None, 1, 1, 10))
        self.real = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 10))
        self.Training = tf.placeholder(dtype=tf.bool)

        # 运行生成网络
        self.G_noise = self.Generator(self.noise, self.labels, self.Training)
        # 运行判别网络
        self.D_real, self.D_real_logits = self.Discriminator(self.x, self.real, self.Training)
        self.D_fake, self.D_fake_logits = self.Discriminator(self.G_noise, self.real, self.Training, reuse=True)
        # 计算每个网络的损失函数
        # 算判别器真值的损失函数
        self.Dis_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
        # 算判别器噪声生成图片的损失函数
        self.Dis_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
        # 损失函数求和
        self.Dis_loss = self.Dis_loss_real + self.Dis_loss_fake
        # 计算生成器的损失函数
        self.Gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))
        # 提取每个网络的变量
        tf_vars = tf.trainable_variables()
        self.Dis_vars = [var for var in tf_vars if var.name.startswith('Discriminator')]
        self.Gen_vars = [var for var in tf_vars if var.name.startswith('Generator')]
        # 调整参数 设计是用来控制计算流图的，给图中的某些计算指定顺序
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # 寻找全局最优点的优化算法，引入了二次方梯度校正 衰减率0.5
            optim = tf.train.AdamOptimizer(self.lr, beta1=0.5)
            # 优化更新训练的模型参数，也可以为全局步骤(global step)计数
            self.D_optim = optim.minimize(self.Dis_loss, global_step=self.global_step, var_list=self.Dis_vars)
            # 寻找全局最优点的优化算法，引入了二次方梯度校正 衰减率0.5
            self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.Gen_loss, var_list=self.Gen_vars)

        tf.global_variables_initializer().run()  # 初始化网络变量
        self.saver = tf.train.Saver()  # 模型保存器

    def restoreModel(self, modelpath=''):
        saver = self.saver
        sess = self.sess
        # 恢复模型
        if modelpath == '':
            print(r'未使用已经保存的模型，从头开始训练。。。！！！')
        else:
            saver.restore(sess, modelpath)
            print(r'模型已经恢复，继续训练。。。！！！')

    def trainModel(self):
        batch_size = self.batch_size
        IMAGE_SIZE = self.IMAGE_SIZE
        onehot = self.onehot
        x = self.x
        noise = self.noise
        labels = self.labels
        real = self.real
        Training = self.Training
        sess = self.sess
        saver = self.saver

        # 对MNIST做一下处理
        train_set = (self.mnist.train.images - 0.5) / 0.5
        train_label = self.mnist.train.labels

        for i in range(self.step):
            Gen_losses = []
            Dis_losses = []
            i_start_time = time.time()
            index = random.sample(range(0, train_set.shape[0]), train_set.shape[0])
            new_set = train_set[index]
            new_label = train_label[index]
            for j in range(new_set.shape[0] // batch_size):
                # 对判别器进行更新
                x_ = new_set[j * batch_size:(j + 1) * batch_size]
                label_ = new_label[j * batch_size:(j + 1) * batch_size].reshape([batch_size, 1, 1, 10])
                real_ = label_ * np.ones([batch_size, IMAGE_SIZE, IMAGE_SIZE, 10])
                noise_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
                loss_d_, _ = sess.run([self.Dis_loss, self.D_optim],
                                      {x: x_, noise: noise_, real: real_, labels: label_, Training: True})
                # 对生成器进行更新
                noise_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
                y_ = np.random.randint(0, 9, (batch_size, 1))
                label_ = onehot[y_.astype(np.int32)].reshape([batch_size, 1, 1, 10])
                real_ = label_ * np.ones([batch_size, IMAGE_SIZE, IMAGE_SIZE, 10])
                loss_g_, _ = sess.run([self.Gen_loss, self.G_optim],
                                      {noise: noise_, x: x_, real: real_, labels: label_, Training: True})
                # 计算训练过程中的损失函数
                errD_fake = self.Dis_loss_fake.eval({noise: noise_, labels: label_, real: real_, Training: False})
                errD_real = self.Dis_loss_real.eval({x: x_, labels: label_, real: real_, Training: False})
                errG = self.Gen_loss.eval({noise: noise_, labels: label_, real: real_, Training: False})
                Dis_losses.append(errD_fake + errD_real)
                Gen_losses.append(errG)
                print('判别器损失函数: %.6f, 生成器损失函数: %.6f' % (np.mean(Dis_losses), np.mean(Gen_losses)))
                if j % 10 == 0:
                    pic = self.save_img + '/' + self.name + str(i * new_set.shape[0] // batch_size + j + 1) + '_' + str(
                        i + 1) + '.png'
                    self.show_result((i + 1), save=True, path=pic)

            # 训练完一轮保存模型
            saver.save(sess, os.path.join(self.save_dir, self.checkpoint_name), global_step=self.global_step)
            print('本次第{0}轮训练完成，模型已经保存！！！！'.format(i))
        # sess.close()

    def getVeriCode(self, path='test.jpg', len = 4):
        resNum = random.randint(0, 9)  # 产生随机数字
        strNum = str(resNum)           # 转成str
        resImg = oneNet.showi(resNum)  # 产生图片
        for i in range(len - 1):
            resNum = random.randint(0, 9)  # 产生数字 str类型
            strNum += str(resNum)
            resImg = np.hstack((resImg, oneNet.showi(resNum)))

        cv2.imwrite(path, resImg)  # 结果图片 resImg
        print('res is : {0}'.format(strNum))  # 结果数字 strNum

if __name__ == '__main__':
    oneNet = ConvoGan()
    oneNet.createModel()
    oneNet.restoreModel(modelpath=r'.\model\train.ckpt-12650')

    # oneNet.trainModel()
    # oneNet.trainModel(modelpath=r'.\model\train.ckpt-550')
    oneNet.getVeriCode(len=7)
