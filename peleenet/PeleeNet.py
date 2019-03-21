# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:33:53 2019
PeleeNet
@author: ThinkPad
"""
from keras.layers import Conv2D,BatchNormalization,Activation,Concatenate,Dense
from keras.layers import MaxPooling2D,Add,Input,ZeroPadding2D,AveragePooling2D,GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model

def Conv_bn_relu(inp, oup, kernel_size=3, stride=1, pad=1,use_relu = True):
    if pad !=0 :
        x=ZeroPadding2D(padding=(pad,pad))(inp)
    else:
        x=inp
    
    x=Conv2D(oup,kernel_size=(kernel_size,kernel_size),strides=(stride,stride),
                 use_bias=False,padding='valid')(x)
    x=BatchNormalization()(x)
    if use_relu:
        x=Activation('relu')(x)
    return x



def res_block(inp,planes=128):
    stem1=Conv_bn_relu(inp,planes,1,1,0)
    stem1=Conv_bn_relu(stem1,planes,3,1,1)
    stem1=Conv_bn_relu(stem1,planes*2,1,1,0,False)
    
    stem2=Conv_bn_relu(inp,planes*2,1,1,0,False)
    out=Add()([stem1,stem2])
    return Activation('relu')(out)

def StemBlock(inp,num_init_features=32):
    stem1=Conv_bn_relu(inp, oup=num_init_features, kernel_size=3, stride=2, pad=1,use_relu = True)
    stem2=Conv_bn_relu(stem1, oup=int(num_init_features/2), kernel_size=1, stride=1, pad=0,use_relu = True)
    stem2=Conv_bn_relu(stem2, oup=num_init_features, kernel_size=3, stride=2, pad=1,use_relu = True)
    stem3=MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(stem1)
    out=Concatenate(axis=-1)([stem2,stem3])
    out=Conv_bn_relu(out, oup=num_init_features, kernel_size=1, stride=1, pad=0,use_relu = True)
    return out
    

def DenseBlock(inp,inter_channel,growth_rate):
    cb1_a=Conv_bn_relu(inp,inter_channel,1,1,0)
    cb1_b=Conv_bn_relu(cb1_a,growth_rate,3,1,1)
    
    cb2_a=Conv_bn_relu(inp,inter_channel,1,1,0)
    cb2_b=Conv_bn_relu(cb2_a,growth_rate,3,1,1)
    cb2_c=Conv_bn_relu(cb2_b,growth_rate,3,1,1)
    return Concatenate(axis=-1)([inp,cb1_b,cb2_c])

def TransitionBlock(inp, oup,with_pooling= True):
    x=Conv_bn_relu(inp,oup,1,1,0)
    if with_pooling:
        x=AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
    return x


def _make_dense_transition(inputs,half_growth_rate,total_filter, inter_channel, ndenseblocks,with_pooling= True):
    x=inputs
    for i in range(ndenseblocks):
        x=DenseBlock(x,inter_channel,half_growth_rate)
    x=TransitionBlock(x,total_filter,with_pooling)
    return x


def PeleeNet(input_shape,num_classes=1000, num_init_features=32,growthRate=32,
             nDenseBlocks = [3,4,8,6], bottleneck_width=[1,2,4,4]):
    inter_channel =list()
    total_filter =list()
    half_growth_rate = int(growthRate / 2)
    
    inp=Input(shape=input_shape)
    stage=StemBlock(inp,num_init_features)
    print(stage.shape)
    for i,b_w in enumerate(bottleneck_width):
        inter_channel.append(int(half_growth_rate*b_w/4)*4)
        if i==0:
            total_filter.append(num_init_features + growthRate * nDenseBlocks[i])           
        else:
            total_filter.append(total_filter[i-1] + growthRate * nDenseBlocks[i])

        if i == len(nDenseBlocks)-1:
            with_pooling = False
        else:
            with_pooling = True
        stage=_make_dense_transition(stage,half_growth_rate,total_filter[i],inter_channel[i],nDenseBlocks[i],with_pooling=with_pooling)
        print(stage.shape)
    out=GlobalAveragePooling2D()(stage)
    out=Dense(1000,activation='softmax')(out)
    model=Model(inputs=inp,outputs=out)
    return model
            
model=PeleeNet((224,224,3)) 
print(model.summary())
plot_model(model, to_file='peleenet.png', show_layer_names=True, show_shapes=True)














