# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:50:54 2023

@author: LJF
"""
from tensorflow.keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from Nets import SCAE, ECNN
import matplotlib.pyplot as plt
import numpy as np


img_size = 128
class_num = 5
BATCH_SIZE = 64
EPOCH = 3
input_shape = (img_size, img_size, 3)
para_scae = [[3], [4,8,16,32]]
#para = [kernel_size, kernel_num]
#kernel_size:       int_list, 每一层的卷积核尺寸,当长度为1时，每一层尺寸相同,不为1时长度必须等于层数, 解码器与编码器相同
#kernel_num:        int_list, 每一层的卷积核个数, 列表长度必须等于层数, 解码器最后一层卷积核数等于输入图像的通道数

source_train_path='.//soure//unlabeled//Train'
source_valid_path='.//soure//unlabeled//Valid'

source_labeled_train_path='.//soure//labeled//Train'
source_labeled_valid_path='.//soure//labeled//Valid'

target_train_path='.//target//Train'
target_valid_path='.//target//Valid'

def plot_result(data, ifsave=False, save_name='plot.png', **kwargs):
    #画曲线
    #data:           ndarray, 曲线的y轴数据, 可以是一维或二维, 二维时的形状(曲线数量, y轴数据)
    #ifsave:         bool, 画图完成后是否保存
    #save_name:      str, 图保存的文件名,必须以'.png'结尾
    #label_list:     str_list, 图例名字列表
    #xlabel:         str, x轴标签
    #ylabel:         str, y轴标签
    curve_num = len(data.shape)
    if curve_num==1:
        data = np.expand_dims(data, axis=0)
    x_len = len(data[0])
    plt.style.use("ggplot")
    plt.figure()
    for i in range(curve_num):
        plt.plot(np.arange(0, x_len), data[i], linestyle='-', label=kwargs['label_list'][i])
    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.legend(loc="best")
    if ifsave:
        plt.savefig(save_name)
    plt.show()
    return True

#Creating a Callback subclass that stores each epoch prediction
class prediction_history(Callback):
    def __init__(self,model):
        self.model=model
        self.acc_batch=[]
        self.loss_batch=[]
    def on_batch_end(self, batch, logs={}):
        self.acc_batch.append(logs.get('accuracy'))
        self.loss_batch.append(logs.get('loss'))

def dataset_create(path, mode):
    #创建数据集
    #path:      str, 数据路径
    #mode:      str, 数据加载的模型, 可选:['input':输入和输出完全相同,用于自监督任务;'categorical':根据目录划分标签,用于分类任务]
    data_datagen = ImageDataGenerator()
    data_generator = data_datagen.flow_from_directory(
        path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode=mode)
    return data_generator


def model_train(model, data_train, data_valid, ifsave=False, save_name='model.h5'):
    #模型训练
    #model:          Model, 完成编译的keras模型
    #data_train：    ImageDataGenerator, 训练集数据
    #data_valid:     ImageDataGenerator, 验证集数据
    #ifsave:         bool, 模型训练完成后是否保存
    #save_name:      str, 模型保存的文件名,必须以'.h5'结尾
    callback=prediction_history(model=model)
    H=model.fit(
         data_train,
         steps_per_epoch=data_train.n/BATCH_SIZE,
         epochs=EPOCH,
         validation_data=data_valid,
         validation_steps=data_valid.n/BATCH_SIZE,
         callbacks=[callback])
    acc_batch = np.array(callback.acc_batch)
    loss_batch = np.array(callback.loss_batch)
    if ifsave:
        model.save(save_name)
    return model, [H, acc_batch, loss_batch]


def unsupervised_learning():
    #无监督学习, 使用无标签源域数据进行自监督训练
    scae = SCAE(input_size = input_shape, para = para_scae)
    scae.compile(optimizer='Adam', loss=['mse'])
    #loading data
    source_data_train = dataset_create(source_train_path, mode = 'input')
    source_data_valid = dataset_create(source_valid_path, mode = 'input')
    #training model
    scae, H_scae = model_train(scae, source_data_train, source_data_valid, ifsave=True, save_name='scae.h5')
    return scae, H_scae

def first_transfer(scae):
    #first transfer
    pre_ecnn = ECNN(input_size = input_shape, out_num=class_num, para = para_scae)
    pre_ecnn.compile(optimizer='Adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
    layer_num = len(para_scae[1])
    for i in range(layer_num):
        layer = pre_ecnn.get_layer('layer_'+str(i))
        encoder = scae.get_layer('encoder_'+str(i))
        weights_en = encoder.get_weights()
        layer.set_weights(weights_en)
    #pre_train ECNN, 使用带标签的源域数据, 进行第一次transfer的pre-train
    source_data_labeled_train = dataset_create(source_labeled_train_path, mode = 'categorical')
    source_data_labeled_valid = dataset_create(source_labeled_valid_path, mode = 'categorical')
    pre_ecnn, H_pre_ecnn = model_train(pre_ecnn, source_data_labeled_train, source_data_labeled_valid, ifsave=True, save_name='pre_ecnn.h5')  
    return pre_ecnn, H_pre_ecnn
    

def second_transfer(pre_ecnn):
    target_data_train = dataset_create(target_train_path, mode = 'categorical')
    target_data_valid = dataset_create(target_valid_path, mode = 'categorical')
    fine_ecnn, H_fine_ecnn = model_train(pre_ecnn, target_data_train, target_data_valid, ifsave=True, save_name='fine_ecnn.h5')  
    return fine_ecnn, H_fine_ecnn

if __name__ == '__main__':
    #H=[history, acc_batch, loss_batch]
    #history        dict, 由fit自动生成,记录每个epoch训练集和验证集最终的acc和loss,
                            #acc=history['accuracy'], val_acc=history['val_accuracy']
                            #loss=history['loss'], val_loss=history['val_loss']
    #acc_batch      list, 由回调函数记录,每个batch迭代的训练集准确率
    #loss_batch     list, 由回调函数记录,每个batch迭代的loss      
    scae, H_scae = unsupervised_learning()                
    pre_ecnn, H_pre_ecnn = first_transfer(scae)
    fine_ecnn, H_fine_ecnn = second_transfer(pre_ecnn)
    #plot
    plot_result(H_pre_ecnn[1], xlabel='Batch', ylabel='Accuracy', label_list=['Pre_ECNN'])
    plot_result(H_pre_ecnn[2], xlabel='Batch', ylabel='Loss', label_list=['Pre_ECNN'])
    plot_result(H_fine_ecnn[1], xlabel='Batch', ylabel='Accuracy', label_list=['Fine_ECNN'])
    plot_result(H_fine_ecnn[2], xlabel='Batch', ylabel='Loss', label_list=['Fine_ECNN'])
