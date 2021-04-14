# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:10:05 2020

@author: sc
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
import cv2

def batch_generator(X, y, batch_size=32, 
                    shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])

class ConvNN(object):
    def __init__(self,batchsize=12, epochs=1000,learning_rate=1e-4,dropout_rate=0.6,shuffle=True,random_seed=None):#初始化
        np.random.seed(random_seed)
        self.batchsize=batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        self.losslog=[]
        
        g = tf.Graph()
        self.merged=None

        with g.as_default():
            ## set random-seed:
            tf.set_random_seed(random_seed)

            ## build the network:
            self.build()

            ## initializer
            self.init_op = tf.global_variables_initializer()

            ## saver
            self.saver = tf.train.Saver()
            
        ## create a session
        #config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(graph=g)#config=config,
        self.writername = tf.summary.FileWriter("./cnn/log",self.sess.graph)    


    def build(self):#网络层
        #kernel：5，strides：2的conv2d定义
        def conv_layer(input,size_in,size_out,kernel_size=3,stridesize=[1,2,2,1],name="conv",use_two_conv=False):
            with tf.name_scope(name):
                w=tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.1),name="W")
                b=tf.Variable(tf.constant(0.1,shape=[size_out]),name="B")
                conv=tf.nn.conv2d(input,w,strides=stridesize,padding="SAME")
                act=tf.nn.relu(conv+b)
                if use_two_conv==True:
                    w=tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_out,size_out],stddev=0.1),name="W_1")
                    b=tf.Variable(tf.constant(0.1,shape=[size_out]),name="B_1")
                    conv=tf.nn.conv2d(input,w1,strides=[1,1,1,1],padding="SAME")
                    act=tf.nn.relu(conv+b1)
                tf.summary.histogram("weights",w)
                tf.summary.histogram("biases",b)
                tf.summary.histogram("activations",act)
                return tf.nn.max_pool(act,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        #全连接层定义
        def fc_layer(input,channels_in,channels_out,name="fc"):
            with tf.name_scope(name):
                w=tf.Variable(tf.truncated_normal([channels_in,channels_out],stddev=0.1),name="W")
                b=tf.Variable(tf.constant(0.1,shape=[channels_out]),name="B")
                return (tf.matmul(input,w)+b)

        ## Placeholders for X and y:#占位符
        tf_x = tf.placeholder(tf.float32, 
                              shape=[None, 24576],  #原784=28*28
                              name='tf_x')
        tf_y = tf.placeholder(tf.int32, 
                              shape=[None],
                              name='tf_y')
        
        is_train = tf.placeholder(tf.bool, 
                              shape=(),
                              name='is_train')

        ## reshape x to a 4D tensor: 
        ## [batchsize, width, height, 1]
        tf_x_image = tf.reshape(tf_x, shape=[-1, 128, 192, 1], 
                              name='input_x_2dimages')
        ## One-hot encoding:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=4,
                              dtype=tf.float32,
                              name='input_y_onehot')

        ## 1st layer: Conv_1
        ## 包含MaxPooling 
        h1 = conv_layer(tf_x_image,1,16,kernel_size=5, name="conv1")
        
        ## 2n layer: Conv_2
        h2 = conv_layer(h1,16,32,kernel_size=3, name="conv2")
         
        ## Dropout
        h2_drop = tf.layers.dropout(h2, 
                              rate=self.dropout_rate,
                              training=is_train)

        ## 3n layer: Conv_3
        h3 = conv_layer(h2_drop,32,64,stridesize=[1,1,1,1], name="conv3")
        
        ## 4n layer: Conv_3
        h4 = conv_layer(h3,64,128,stridesize=[1,1,1,1], name="conv4")
        
        #GAP层
        h5 = tf.keras.layers.GlobalAveragePooling2D(name="GAP")(h4)
        input_shape = h5.get_shape().as_list()[-1]
        
        ## 4th layer: Fully Connected (linear activation)
        h6 = fc_layer(h5,input_shape,4,name="fc1")

        ## Prediction
        predictions = {
            'probabilities': tf.nn.softmax(h6, 
                              name='probabilities'),
            'labels': tf.cast(tf.argmax(h6, axis=1), 
                              tf.int32, name='labels')}
        
        ## Loss Function and Optimization
        l2_norm=tf.nn.l2_loss(
                tf.nn.softmax_cross_entropy_with_logits(
                logits=h6, labels=tf_y_onehot))
        loss_mean=tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                logits=h6, labels=tf_y_onehot))
        self.cross_entropy_loss = l2_norm+loss_mean
        
        ## Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(self.cross_entropy_loss,
                                       name='train_op')

        ## Finding accuracy
        correct_predictions = tf.equal(
            predictions['labels'], 
            tf_y, name='correct_preds')
        
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name='accuracy')
        
        self.merged = tf.summary.merge_all()#将tensorboard需要加载的变量整合    

    def save(self, epoch, path='./cnn/tflayers-model'):#存模型
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Saving model in %s' % path)
        self.saver.save(self.sess, 
                        os.path.join(path, 'model.ckpt'),
                        global_step=epoch)
        
    def load(self, epoch, path):#加载模型
        print('Loading model from %s' % path)
        self.saver.restore(self.sess, 
             os.path.join(path, 'model.ckpt-%d' % epoch))


    def train(self, training_set, 
              validation_set=None,
              initialize=True):
        ## initialize variables
        if initialize:
            self.sess.run(self.init_op)
            
        self.train_cost_ = []
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])#加载完整的训练数据集，数据加标签

        tensorboard_tradir = './cnn/log'

        if not os.path.exists(tensorboard_tradir):
            os.makedirs(tensorboard_tradir)
        
        self.writername = tf.summary.FileWriter(tensorboard_tradir)
        self.writername.add_graph(self.sess.graph)#将图写入tensorboard

        def evaluate(validation_set,epoch):
            '''批量的形式计算验证集或测试集上数据的平均loss，平均accuracy'''
            data_len = 0
            valid_cost = 0.0
            X_data_valid = np.array(validation_set[0])
            y_data_valid = np.array(validation_set[1])#加载测试数据集

            batch_gen_valid =batch_generator(X_data_valid, y_data_valid ,batch_size=100)
            total_acc = 0.0
            for i, (batch_x,batch_y) in enumerate(batch_gen_valid):
                batch_len=batch_x.shape[0]
                data_len=data_len+1
                feed = {'tf_x:0': batch_x.reshape((batch_len,-1)),
                        'tf_y:0': batch_y,
                        'is_train:0': False} ## for dropout
                loss, acc = self.sess.run(
                    [self.cross_entropy_loss, self.accuracy], 
                    feed_dict=feed)
                valid_cost += loss
                total_acc += acc
    
            return valid_cost/data_len,total_acc/data_len

        for epoch in range(1, self.epochs + 1):
            if epoch==1001:
                self.learning_rate=2e-5#进一步缩小学习率，1001根据tensorboard中acc的趋势确定
            batch_gen = batch_generator(X_data, y_data,batch_size=self.batch_size, 
                                 shuffle=self.shuffle)
            avg_loss = 0.0
            avg_acc_train=0.0
            data_len=0
            for i, (batch_x,batch_y) in enumerate(batch_gen):
                data_len=data_len+1
                feed = {'tf_x:0': batch_x.reshape((batch_x.shape[0],-1)),
                        'tf_y:0': batch_y,
                        'is_train:0': True} ## for dropout
                loss, train_acc,summary_info, _ = self.sess.run(
                        [self.cross_entropy_loss, self.accuracy,self.merged,'train_op'], 
                        feed_dict=feed)
                avg_loss += loss
                avg_acc_train+=train_acc

            avg_loss=avg_loss/data_len
            avg_acc_train=avg_acc_train/data_len

            print('Epoch %02d: Training Avg. Loss: '
                  '%7.2f Training Acc: %7.3f' % (epoch, avg_loss,avg_acc_train), end=' ')
            summary.value.add(tag='train_acc', simple_value=avg_acc_train)   
            summary.value.add(tag='train_loss', simple_value=avg_loss)

            if validation_set is not None:
                cost,acc=evaluate(validation_set,epoch)
                summary = tf.Summary() 
                summary.value.add(tag='valid_acc', simple_value=acc)
                summary.value.add(tag='valid_loss', simple_value=cost)
                print('Validation Acc: %7.3f ' % acc)

            else:
                print()
            self.writername.add_summary(summary_info,epoch)
            self.writername.add_summary(summary,epoch)                

                
    def predict(self, X_test, return_proba = False):#用已训练模型预测
        
        feed = {'tf_x:0': X_test.reshape((X_test.shape[0],-1)),
                'is_train:0': False} ## for dropout
        if return_proba:
            return self.sess.run('probabilities:0',
                                 feed_dict=feed)#返回概率
        else:
            return self.sess.run('labels:0',
                                 feed_dict=feed)#返回标签

# In[33]:
import os
import cv2
import random
#数据加载与训练集划分
def duqu(X,y,path):#读取路径下所有文件，与输入的X,y矩阵拼接
    filenames=os.listdir(path)
    for filename in filenames:
        f0=os.path.splitext(filename)[0]
        label=f0.split('_',1)[0]#取样本标签
        y.append(int(label))
        im=(cv2.imread(path+filename,0)).astype(float)#黑白图，无通道维数
        im=im.reshape(1,128,192,1)
        X=np.append(X,im,axis=0)
    return X,y
def duqu_2(X,y,path):#读取路径下所有文件,而后构建不平衡样本集
    filenames=os.listdir(path)
    y22=np.array([])
    y11=[]
    X11=np.empty([1,128,192,1])
    X22=np.empty([1,128,192,1])
    size=200
    for filename in filenames:
        f0=os.path.splitext(filename)[0]
        label=int(f0.split('_',1)[0])
        y11.append(int(label))
        im=(cv2.imread(path+filename,0)).astype(float)#黑白图，无通道维数
        im=im.reshape(1,128,192,1)
        X11=np.append(X11,im,axis=0)
    X11=X11[1:,:,:,:]#完整读取路径下所有文件
    for i in range(1,4):#读取1-3标签下的size个样本
        y11=np.array(y11)
        idx=np.where(y11==i)[0]
        idx_noise=random.sample(list(idx),size)
        y22=np.append(y22,y11[idx_noise],axis=0)
        X22=np.append(X22,X11[idx_noise,:,:,:],axis=0)
    X22=X22[1:,:,:,:]
    return  np.append(X,X22,axis=0),np.append(y,y22,axis=0)    
            
def ratio_dutu(y,ratio=0.2,size_0=500):#部分读取路径下文件，同样是构建不平衡样本集
    def xunhuan(label_du,y):
        import random
        idx_du=np.where(y==label_du)[0]
        if label_du==0:
            idx_qutu_du=random.sample(list(idx_du),size_0)
        else:
            idx_qutu_du=random.sample(list(idx_du),int(size_0*ratio))
        return idx_qutu_du
    idx=np.array([],dtype='int32')
    for i in range(4):
        idx=np.append(idx,xunhuan(i,y))
    return idx

path1='G://wind-energy//windpower//data//train//'
y=[]
X=np.empty([1,128,192,1])
X,y=duqu(X,y,path1)
X=X[1:,:,:,:]#因为有一个empty，需要把其删去
y1=np.array(y)
np.save('./cnn/data/XX.npy',X)
np.save('./cnn/data/yy1.npy',y1)

#X=np.load('./cnn/data/proposedtry/X.npy')
#y1=np.load('./cnn/data/proposedtry/y1.npy')

#获取不平衡数据集
idx=ratio_dutu(y1,ratio=0.55,size_0=500)
X=X[idx,:,:,:]
y1=y1[idx]

#混合增强的生成样本
path2='G://wind-energy//windpower//sngan_projection//sngan_projection_TensorFlow-master//resultsimages//11//'
X,y=duqu(X,list(y1),path2)
y1=np.array(y)#平衡后样本集
np.save('./cnn/data/proposedtry/X.npy',X)
np.save('./cnn/data/proposedtry/y1.npy',y1)

#混合降噪后样本集
'''
path3='G://wind-energy//windpower//chapter5//DnCNN//sharpen//'
y1=[]
X=np.empty([1,128,192,1])
X,y1=duqu(X,y1,path3)
X=X[1:,:,:,:]#因为有一个empty，需要把其删去
y1=np.array(y1)
np.save('./cnn/data/chapter5/X_tu1.npy',X)
np.save('./cnn/data/chapter5/y1_tu1.npy',y1)

X2=np.load('./cnn/data/chapter5/X_tu1.npy')
y2=np.load('./cnn/data/chapter5/y1_tu1.npy')
y1=np.append(y2,y1,axis=0)
X=np.append(X2,X,axis=0)
#由原始样本集与降噪后样本集混合作为测试集，两者数量相同
X3=np.load('./cnn/data/XX.npy')
y3=np.load('./cnn/data/yy1.npy')
idx=ratio_dutu(y3,ratio=0.55,size_0=500)
X3=X3[idx,:,:,:]
y3=y3[idx]
te_y=np.append(y2,y3,axis=0)
te_X=np.append(X2,X3,axis=0)
'''
#分析含噪声样本集对模型诊断能力影响
#X=np.load('./cnn/data/to_balance/4/X_addnoise.npy')
#y1=np.load('./cnn/data/to_balance/4/y1_addnoise.npy')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y1, random_state=0,train_size=0.7,test_size=0.3,stratify=y1) #相同占比划分测试集与训练集

'''
#原始故障数据集te_X 与 te_y(均为array),用于测试平衡前后模型acc
path4='G://wind-energy//windpower//data//4//train//'
filenames=os.listdir(path4)
te_size=1000
te_y=[]
te_X=np.empty([1,128,192,1])
for filename in filenames:
    f0=os.path.splitext(filename)[0]
    label=int(f0.split('_',1)[0])
    #if not label == 0:
    te_y.append(label)
    im=(cv2.imread(path4+filename,0)).astype(float)#黑白图，无通道维数
    im=im.reshape(1,128,192,1)
    te_X=np.append(te_X,im,axis=0)
te_X=te_X[1:,:,:,:]
zongshu=len(te_y)
te_idx=random.sample(range(0,zongshu),te_size)
te_y=np.array(te_y)
te_y=te_y[te_idx]
te_X=te_X[te_idx,:,:,:]
np.save('./cnn/4/purefault_test/te_X_balance.npy',te_X)
np.save('./cnn/4/purefault_test/te_y_balance.npy',te_y)
'''
te_X=np.load('./cnn/data/XX.npy')
te_y=np.load('./cnn/data/yy1.npy')#完整样本集作测试集

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
np.save('./cnn/data/mean_vals_gan.npy',mean_vals)
np.save('./cnn/data/std_val_gan.npy',std_val)#加载模型预测时，最好将mean与std同样加载
#mean_vals = np.load('./cnn/data/chapter5/mean_vals.npy')
#std_val = np.load('./cnn/data/chapter5/std_val.npy')
X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val
te_X_centered = (te_X - mean_vals)/std_val
del X_train, X_test

#训练
cnn = ConvNN(random_seed=123)
cnn.train(training_set=(X_train_centered, y_train), 
      validation_set=(X_valid_centered, y_valid))#若是加载模型后再训练，先load后train，且initialize=False

cnn.save(epoch=1000)
cnn.load(epoch=1000, path='./cnn/tflayers-model/1')
y=cnn.predict(te_X_centered)

#tensorboard启动命令
#tensorboard --logdir=./cnn/log

#混淆矩阵绘制
def huatu(preds,te_y,eepoch):
    import seaborn
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    plt.rcParams['savefig.dpi'] = 300
    con_mat2 = confusion_matrix(te_y, preds)
    con_mat_norm2 = con_mat2.astype('float') / con_mat2.sum(axis=1)[:, np.newaxis]     # 归一化
    con_mat_norm2 = np.around(con_mat_norm2, decimals=2)#两位小数
    figure2=plt.figure(figsize=(7,5))
    seaborn.set(font_scale=1.3)
    seaborn.heatmap(con_mat_norm2,annot=True,cmap='Blues')
    #plt.xaxis.tick_top()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.ylabel('真实状态', fontsize=18)
    plt.xlabel('诊断状态', fontsize=18)
    plt.tick_params(axis='y',labelsize=14) # y轴, labelrotation=45
    plt.tick_params(axis='x',labelsize=14) # x轴
    a=(100*np.sum(te_y == preds)/len(te_y))
    plt.savefig('./cnn/4/acc_test/chap5gan_balancewhole_%f_%d.jpg' %(round(a, 2),eepoch))
    plt.show()
    print('Test Accuracy: %.2f%%' % (100*np.sum(te_y == preds)/len(te_y)))

huatu(preds,te_y,1001)
print(cnn2.predict(te_X_centered[:,:]))


