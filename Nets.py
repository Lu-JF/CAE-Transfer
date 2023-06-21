# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:10:10 2023

@author: LJF
"""

from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Flatten, Conv1D, Subtract, Reshape, add, UpSampling2D
from keras.models import Model
from numpy import ones

def DeConv2D(net_input, kernel_num, size, name):
    #反卷积, 先进行上采样2倍, 再进行卷积
    #net_input:         网络计算图的输入映射,形状为(:,length,with,channel)
    #kenerl_num:        int, 卷积核数目
    #size:              int, 卷积核尺寸
    net_input = UpSampling2D(size=(2,2))(net_input)
    net_output = Conv2D(filters=kernel_num, kernel_size=size, strides=1, padding='same', activation='selu',  name = name)(net_input)
    return net_output


def Covolutional_Encoder(net_input, layer_num, size, kernel_num, code_num):
    #卷积编码器
    #net_input:         网络计算图的输入映射,形状为(:,length,with,channel)
    #layer_num:         int, 编码器的总层数
    #size:              int_list, 每一层的卷积核尺寸,当长度为1时，每一层尺寸相同,不为1时长度必须等于层数
    #kernel_num:        int_list, 每一层的卷积核个数, 列表长度必须等于层数
    #code_num:          int, 编码器最终编码的神经元数目
    #net_output:        网络计算图的输出映射,形状为(:,code_num)
    #code2d_shape:      编码2维形态的形状
    if len(size)!=1 and len(size)!=layer_num:
        return False, False
    for i in range(layer_num):
        if len(size)==1:
            net_input = Conv2D(filters=kernel_num[i], kernel_size=size[0], strides=1, padding='same', activation='selu', name = 'encoder_'+str(i))(net_input)
        else:
            net_input = Conv2D(filters=kernel_num[i], kernel_size=size[i], strides=1, padding='same', activation='selu', name = 'encoder_'+str(i))(net_input)
        net_input = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(net_input)
    code2d_shape = net_input.shape[1:]
    net_input = Flatten()(net_input)
    net_output = Dense(code_num)(net_input)
    return net_output, code2d_shape


def Convolutional_Decoder(net_input, layer_num, size, kernel_num, code2d_shape):
    #卷积解码器
    #net_input:         网络计算图的输入映射,形状为(:,code_num)
    #layer_num:         int, 解码器的总层数
    #size:              int_list, 每一层的卷积核尺寸,当长度为1时，每一层尺寸相同,不为1时长度必须等于层数
    #kernel_num:        int_list, 每一层的卷积核个数, 列表长度必须等于层数
    #code_2dshape:      tuple, 编码二维形态的形状
    #net_output:        网络计算图的输出映射,形状为(:,code_num)
    if len(size)!=1 and len(size)!=layer_num:
        return False, False
    net_input = Dense(code2d_shape[0]*code2d_shape[1]*code2d_shape[2])(net_input)
    net_input = Reshape(code2d_shape)(net_input)
    for i in range(layer_num):
        if len(size)==1:
            net_input = DeConv2D(net_input, kernel_num[i], size[0], name = 'decoder_'+str(i))
        else:
            net_input = DeConv2D(net_input, kernel_num[i], size[i], name = 'decoder_'+str(i))
    net_output = net_input
    return net_output

def SCAE(input_size, para):
    #stacked convolutional auto encoder
    #para = [layer_num, kernel_size, kernel_num]
    #layer_num:         int, 编码器的总层数, 编码器和解码器层数相同
    #kernel_size:       int_list, 每一层的卷积核尺寸,当长度为1时，每一层尺寸相同,不为1时长度必须等于层数, 解码器与编码器相同
    #kernel_num:        int_list, 每一层的卷积核个数, 列表长度必须等于层数, 解码器最后一层卷积核数等于输入图像的通道数
    layer_num = len(para[1])
    kernel_size = para[0]
    encoder_num = para[1]
    decoder_num = encoder_num.copy()
    list.reverse(decoder_num)
    decoder_num[-1] = input_size[-1]
    
    net_input=Input(shape=input_size)
    code, shape_2d = Covolutional_Encoder(net_input, layer_num=layer_num, size=kernel_size, kernel_num=encoder_num, code_num=100)
    net_output = Convolutional_Decoder(code, layer_num=layer_num, size=kernel_size, kernel_num=decoder_num, code2d_shape=shape_2d)
    model = Model(inputs=net_input, outputs=net_output)
    return model

def ECNN(input_size, out_num, para):
    layer_num=len(para[1])
    size=para[0]
    kernel_num=para[1]
    model_input=Input(shape=input_size)
    net_input = model_input
    for i in range(layer_num):
        net_input = Conv2D(filters=kernel_num[i], kernel_size=size[0], strides=1, padding='same', activation='selu', name = 'layer_'+str(i))(net_input)
        net_input = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(net_input)
    net_input = Flatten()(net_input)
    net_input = Dense(100)(net_input)
    model_out= Dense(out_num, activation='softmax')(net_input)
    model = Model(inputs=model_input, outputs = model_out)
    return model
    
if __name__ == '__main__':
    input_shape = (256,256,1)
    para_scae = [[3], [4,8,16,32]]
    scae = SCAE(input_size = input_shape, para = para_scae)
    scae.compile(optimizer='Adam', loss=['mse'])
    ecnn = ECNN(input_size = input_shape, out_num = 10, para = para_scae)
    ecnn.compile(optimizer='Adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
