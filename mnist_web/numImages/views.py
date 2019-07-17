from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

import numpy as np
import json
from django.http import HttpResponse
import cv2
from django.views.decorators.csrf import csrf_exempt
import base64
from convo_gan_ooad import ConvoGan

# Create your views here.


def receive_num(request):
    if request.GET:
        num = request.GET.get('input', None)
        print(type(num))
        if num:
            num = int(num)
            oneNet = ConvoGan()
            oneNet.createModel()
            oneNet.restoreModel(modelpath='./model/train.ckpt-12650')
            oneNet.showi(num, path='./static/num.jpg')

            return render(request, 'get_num.html', {'image': 'num.jpg', })
        else:
            return render(request, 'get_num.html', {'image': 'timg.gif', })
    else:
        return render(request, 'get_num.html', {'image': 'timg.gif',  })

def receive_code(request):
    if request.GET:
        code = request.GET.get('incode', None)
        print(type(code))
        print(code)
        if code != '':
            num = int(code)
            if num == 5318:
                return HttpResponse("验证成功！")
            else:
                return HttpResponse(code.html,'验证失败')
            # oneNet = ConvoGan()
            # oneNet.createModel()
            # oneNet.restoreModel(modelpath='./model/train.ckpt-12650')
            # oneNet.showi(num, path='./static/cpde.png')
            # return render(request, 'code.html', {'codeimage': 'code.png', })
        else:
            return render(request, 'code.html', {'codeimage': 'code.png', })
    else:
        return render(request, 'code.html', {'codeimage': 'null.png', })
# def get_valid_img(request):
#     # 不需要在硬盘上保存文件，直接在内存中加载就可以
#     io_obj = BytesIO()
#     # 将生成的图片数据保存在io对象中
#     img_obj.save(io_obj, "png")
#     # 从io对象里面取上一步保存的数据
#     data = io_obj.getvalue()
#     return HttpResponse(data)
def shibie(request):
    return render(request, 'index2.html', locals())

@csrf_exempt
def scene(request):
    print("sdf")
    if request.method == "POST":
        print("request.method == post")
        name = request.POST.get('fileName', None)
        data = request.POST.get('fileData')

        # print(len(data))

        missing_padding = 4 - len(data) % 4
        # print(data)

        # for i in range(missing_padding+1):
        #     data = data[:-1]
        data = data[22:]
        imagedata = base64.b64decode(data)
        # dy = "="
        #
        # if missing_padding:
        #     data += dy * missing_padding
        #     #return base64.decodestring(data)
        #     imagedata = base64.b64decode(data)
        # else:
        #     imagedata = base64.b64decode(data)

        # print(imagedata)

        file = open('pic.png', "wb")
        file.write(imagedata)
        file.close()

        img = cv2.imread('pic.png')
        gray_img = np.zeros((280, 280))

        for i in range(280):
            for j in range(280):
                gray_img[i,j] = img[i,j,2]

        cv2.imwrite('static/gray.png', gray_img)

        # show_img = cv2.resize(gray_img, (140, 140))
        new_img = cv2.resize(gray_img, (28, 28))
        show_img = cv2.resize(new_img, (140, 140))
        new_img = cv2.blur(new_img, (2, 2))
        # print(new_img.shape)
        cv2.imwrite('static/new_gray.png', new_img)
        cv2.imwrite('static/show_gray.png', show_img)

        # w1 = np.loadtxt("data/weight1.txt")
        # w2 = np.loadtxt("data/weight2.txt")
        # w3 = np.loadtxt("data/weight3.txt")
        # b1 = np.loadtxt("data/bios1.txt")
        # b2 = np.loadtxt("data/bios2.txt")
        # b3 = np.loadtxt("data/bios3.txt")

        matrix1 = np.asarray(new_img)
        matrix1 = np.mat(matrix1)
        matrix1 = matrix1.reshape(1, 784)

        # w1 = np.mat(w1)
        # h1 = np.dot(matrix1, w1) + b1
        # h1 = sigmoid(h1)
        #
        # w2 = np.mat(w2)
        # h2 = np.dot(h1, w2) + b2
        # h2 = sigmoid(h2)
        #
        # w3 = np.mat(w3)
        # h3 = np.dot(h2, w3) + b3
        #
        # # print(h3.shape)
        # # print(h3)

        # ans = np.argmax(h3)
        # result = str(ans)

        import tensorflow as tf
        # mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
        # print(type(mnist.train.labels))
        # print(mnist.train.labels.shape)

        def weight_variable(shape):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial)

        def bias_variable(shape):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)

        def conv2d(x, W):
                 return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x = tf.placeholder("float", shape=[None, 784])
        y_ = tf.placeholder("float", shape=[None, 10])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        saver = tf.train.Saver()

        ans = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
        ans = np.array(ans)
        ans = ans.reshape(1, 10)
        # print(type(ans))

        # with tf.Session() as sess2:
        #     # tf.reset_default_graph()
        #     tf.get_variable_scope().reuse_variables()
        #     saver.restore(sess2, './static/model/model.ckpt')
        #     ans2 = sess2.run(y_conv, feed_dict={
        #                 x: matrix1, y_: ans, keep_prob: 1.0})
            # print("test accuracy %g" % y_conv.eval(feed_dict={
            #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        sess2 = tf.Session()
        tf.reset_default_graph()
        saver.restore(sess2, './static/model/model.ckpt')
        ans2 = sess2.run(y_conv, feed_dict={
                            x: matrix1, y_: ans, keep_prob: 1.0})

        print(ans2)
        ans = np.argmax(ans2)
        # print(str(ans2))
        dicc = {}
        for i in range(10):
            dicc[i] = {"num": str(ans2[0, i])}

        # print(dicc)

        return HttpResponse(json.dumps(dicc))
        # return HttpResponse(str(ans2))



