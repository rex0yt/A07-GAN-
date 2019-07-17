import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

import os


class gan_network:

    def __init__(self):
        # 生成模型的输入和参数初始化

        self.G_W1 = tf.Variable(self.xavier_init(size=[100, 128]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.G_W2 = tf.Variable(self.xavier_init(size=[128, 784]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[784]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # 判别模型的输入和参数初始化

        self.D_W1 = tf.Variable(self.xavier_init(size=[784, 128]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.D_W2 = tf.Variable(self.xavier_init(size=[128, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

        # #判别器
        # self.D_INPUT_SIZE = 784
        # self.D_OUTPUT_SIZE = 1
        #
        # #生成器
        # self.G_INPUT_SIZE = 100
        # self.G_OUTPUT_SIZE = 784
        # self.HIDDEN_LAYER = [128]
        self.step = 100000
        self.batch_size = 128
        self.learning_rate = 0.001
        # dropout 参数
        self.keep_prob = tf.placeholder(tf.float32)

        self.X = tf.placeholder(tf.float32, [None, 784])
        # self.D_y_actual = tf.placeholder(tf.float32, [None, self.D_OUTPUT_SIZE])

        self.Z = tf.placeholder(tf.float32, [None, 100])
        # self.G_y_actual = tf.placeholder(tf.float32, [None, self.G_OUTPUT_SIZE])

        self.save_dir = "model"
        self.checkpoint_name = "train.ckpt"
        # self.G_weight = []  #权重
        # self.G_bias = []    #偏置
        #
        # self.D_weight = []  # 权重
        # self.D_bias = []    # 偏置
        # self.sess = tf.Session()


        # 喂入数据
        self.G_sample = self.generator(self.Z)
        self.D_logit_real = self.discriminator(self.X)
        self.D_logit_fake = self.discriminator(self.G_sample)
        # 计算loss
        self.D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real,
                                                                                  labels=tf.ones_like(self.D_logit_real)))
        self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake,
                                                                                  labels=tf.zeros_like(self.D_logit_fake)))
        self.D_loss = self.D_fake_loss + self.D_real_loss

        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake,
                                                                             labels=tf.ones_like(self.D_logit_fake)))
        # theta_D = [self.D_weight + self.D_bias]
        # theta_G = [self.G_weight + self.G_bias]
        # print(self.G_weight)
        # print(self.D_weight)

        self.D_optimizer = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.theta_D)
        self.G_optimizer = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.theta_G)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = tf.sqrt(2. / in_dim)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    # 生成模型：产生数据
    def generator(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        # G_log_prob = tf.nn.dropout(G_log_prob, self.keep_prob)
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob
        # #创建模型
        # tmp = z * 1
        # for i in range(len(self.G_weight)):
        #     tmp = tf.matmul(tmp, self.G_weight[i])+self.G_bias[i]
        #     #使用drop-out正则化
        #     #tmp = tf.nn.dropout(tmp, self.keep_prob)
        #     tmp = tf.nn.sigmoid(tmp)
        # return tmp

    # 判别模型:真实值和概率值
    def discriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        # D_logit = tf.nn.dropout(D_logit, self.keep_prob)
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob
        # #创建模型
        # # print(self.D_weight)
        # # print(self.D_bias)
        # tmp = x * 1
        # for i in range(len(self.D_weight)):
        #     tmp = tf.matmul(tmp, self.D_weight[i])+self.D_bias[i]
        #     #使用drop-out正则化
        #     #tmp = tf.nn.dropout(tmp, self.keep_prob)
        #     tmp = tf.nn.sigmoid(tmp)
        # return tmp

    # 随机噪声产生
    def sample_z(self, m, n):
        return np.random.uniform(-1.0, 1.0, size=[m, n])

    def plot(self, samples, n):
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        n = n / 1000
        plt.savefig('out/' + str(n) + '.png')

    # 模型恢复
    def run(self, mnist):

        tf.global_variables_initializer().run()  # 初始化网络变量
        # # 生成器整体网络模型
        # G_layer = [self.G_INPUT_SIZE]
        # G_layer = G_layer + self.HIDDEN_LAYER
        # G_layer = G_layer + [self.G_OUTPUT_SIZE]
        # # 创建各层权重
        #
        # for i in range(len(G_layer) - 1):
        #     self.G_weight.append(tf.Variable(tf.truncated_normal([G_layer[i], G_layer[i + 1]], stddev=0)))
        #     self.G_bias.append(tf.Variable(tf.constant(0.1, shape=[G_layer[i + 1]])))
        #
        # # 判别器整体网络模型
        # D_layer = [self.D_INPUT_SIZE]
        # D_layer = D_layer + self.HIDDEN_LAYER
        # D_layer = D_layer + [self.D_OUTPUT_SIZE]
        # # 创建各层权重
        # for i in range(len(D_layer) - 1):
        #     self.D_weight.append(tf.Variable(tf.truncated_normal([D_layer[i], D_layer[i + 1]], stddev=0)))
        #     self.D_bias.append(tf.Variable(tf.constant(0.1, shape=[D_layer[i + 1]])))

        # 加载模型
        # with tf.Session() as sess2:
        #     flag = os.path.isfile(self.save_dir + r'/checkpoint')
        #     if flag:
        #         self.saver.restore(sess2, r'.\model\train.ckpt')
        #         print("模型恢复成功")
        #     else:
        #         print("不存在模型，重新训练")
        #         self.train(mnist)

        flag = os.path.isfile(self.save_dir + r'/checkpoint')
        if flag:
            self.saver.restore(self.sess, r'.\model\train.ckpt')
            print("模型恢复成功")
        else:
            print("不存在模型，重新训练")
        self.train(mnist)

    # 训练
    def train(self, mnist):
        sess = self.sess

        # 图片输出文件夹
        if not os.path.exists('out/'):
            os.makedirs('out/')

        # 开始训练
        print("=====================开始训练============================")

        # with tf.Session() as sess:
        for it in range(self.step):
            X_mb, _ = mnist.train.next_batch(batch_size=self.batch_size)
            # print(X_mb)
            _, D_loss_curr = sess.run([self.D_optimizer, self.D_loss],
                                      feed_dict={self.X: X_mb, self.Z: self.sample_z(128, 100), self.keep_prob: 0.5})
            _, G_loss_curr = sess.run([self.G_optimizer, self.G_loss],
                                      feed_dict={self.Z: self.sample_z(128, 100), self.keep_prob: 0.5})
            if it % 1000 == 0:
                print('====================打印出生成的数据============================')
                samples = sess.run(self.G_sample, feed_dict={self.Z: self.sample_z(16, 100), self.keep_prob: 0.5})
                self.plot(samples, it)

                print('iter={}'.format(it))
                print('D_loss={}'.format(D_loss_curr))
                print('G_loss={}'.format(G_loss_curr))
                # 模型保存
                self.saver.save(sess, os.path.join(self.save_dir, self.checkpoint_name))
                print("模型保存成功")


if __name__ == '__main__':
    # 读入数据
    mnist = input_data.read_data_sets('./data', one_hot=True)
    # 实例化
    gan_net = gan_network()
    gan_net.run(mnist)
