############################################################################
# Author            :   ZDF
# Created on        :   02/20/2020 Thu
# last modified     :   02/20/2020 Thu
# Description       :
# 1. basic frame for MISO VLC system: signal constraints not good. espeicially the maximum constraint
# 2.    # two problem here: 1. how to describe None as the dimension of parameter
#     # 2. AttributeError: 'Tensor' object has no attribute 'numpy'
# according to the examples in github, we need to construct the NN via selfdefined layers instead of keras
############################################################################

############################################################################
# ### Import libs
############################################################################
# * using tensorflow2.0(us tf.keras rather than pure keras)
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Nadam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d.axes3d import Axes3D

#cvxpy layer
import cvxpy as cp
from cvxpylayers.tensorflow import CvxpyLayer
#not clear whether is useful
tf.executing_eagerly

import sys
# import keras
import logging
import os
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

############################################################################
# system parameters
############################################################################
# * define (n_channel,k) here for (n,k) autoencoder
M = 8
n_channel = 3  # RGBY
k = np.log2(M)
k = int(k)

R = k / n_channel
print('M:', M, 'k:', k, 'n:', n_channel)
############################################################################
# train parameters
############################################################################
epochs = 10
epochs_switch_snr = 2
batch_size = 512

N_train = 1000000
N_test  = 10000
N_MonteCarlo = 1

# vlc parameter
A = 1  # peak amplitude
SNR_train_dB_1 = np.random.uniform(10,15,size=N_train)

############################################################################
# channel parameters
############################################################################
h =tf.constant([[1] , [2], [3]],dtype=tf.float32)
print("h=",h)
h1 = 1
h2 = 1
h3 = 1

############################################################################
# constraints parameters
############################################################################

for i_color in [0]:
    if (i_color == 0):
        g11=86e4

    elif i_color == 1:
        g11 = 76e4

    ############################################################################
    # Generate one hot encoded vector
    ############################################################################
    # #generating data1 of size M
    label1 = np.random.randint(M, size=N_train)
    data1 = tf.one_hot(
        label1,
        M,
        on_value=1.0,
        off_value=0.0,
        axis=None,
        dtype=None,
        name=None
    )
    data1 = np.array(data1)
    # dd = tf.keras.backend.mean(data1, axis=1)
    # cc = tf.abs(tf.keras.backend.mean(data1, axis=0))
    data1
    # print (data1)
    print (data1.shape)

    # #### defining autoencoder and it's layer
    # ###########################################################
    # TX part
    # ###########################################################
    input_signal = Input(shape=(M,),name='signal_input')
    # encoded_s_combine = Dense(M, activation='relu')(input_signal)
    # print (data1[1,:])
    # encoded_s_combine = Dense(M, activation='linear')(input_signal)
    # print (encoded_s_combine[1,:])
    # encoded_s_combine = Dense(M , activation='linear')(encoded_s_combine)
    # encoded_s_combine = Dense(M , activation='relu')(encoded_s_combine)
    # encoded_s_combine = Dense(M , activation='relu')(encoded_s_combine)
    # encoded_s_combine = BatchNormalization(momentum=0, center=False, scale=False)(encoded_s_combine)
    # print (encoded_s_combine[1,:])
    x_tx = Dense(n_channel, activation='linear',name='hidden_layer_1')(input_signal)

    # ###########################################################
    # TX signal constraints
    # ###########################################################
    # print (encoded_s_combine[1,:])
    # encoded_s_combine = Dense(n_channel, activation='relu')(encoded_s_combine)
    # encoded_s_combine = BatchNormalization(momentum=0, center=False, scale=False)(encoded_s_combine)
    # softplus & hard_sigmoid & exponential (good)
    # relu (bad) linear (wrong)
    # encoded_constraint = Dense(n_channel, activation='elu')(encoded_s_combine) + 1
    # encoded_s_combine = Dense(n_channel, activation='linear',name='hidden_layer_1')(input_signal)
    # encoded_s_combine = tf.keras.activations.elu(encoded_s_combine,alpha=1.0) + 1
    # tf.keras.layers.ELU(alpha=1.0)
    # encoded_s_combine = Dense(n_channel, activation='tanh')(encoded_s_combine)
    # encoded_constraint = encoded_s_combine / tf.math.reduce_max(encoded_s_combine)
    # encoded_constraint = tf.keras.activations.softsign(encoded_s_combine) + 1
    # encoded_constraint = Dense(n_channel, activation='sigmoid')(encoded_s_combine)
    # encoded_constraint = Dense(n_channel, activation='relu')(encoded_s_combine)
    # encoded_constraint = Lambda(lambda x: tf.math.minimum(1.0,tf.math.maximum(encoded_s_combine,0.0)))(encoded_s_combine)
    # encoded_constraint = Lambda(lambda x: tf.math.minimum(1.0,encoded_s_combine),name='max_constraint')(encoded_s_combine)
    # print ("encoded_constraint",encoded_constraint[1,:])

    # total optical power (lm) constraint added to loss
    # total_o_power_train = 190
    # print ("encoded_constraint:",encoded_constraint.shape)
    # mean of batch, sum of all led
    # total_o_power_loss = Lambda(lambda x: tf.expand_dims(tf.abs(tf.reduce_sum(tf.keras.backend.mean(encoded_constraint, axis=0)) - total_o_power_train),axis=0), name='total_o_power_loss')\
    #         (encoded_constraint)

    # signal constraint
    amplitude_constraint = 1
    # mean_constraint = batch_size * tf.constant([1, 2, 3], dtype=tf.float32)
    mean_constraint = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    x_example =tf.constant([1, 2, 3],dtype=tf.float32)
    _x = cp.Parameter((n_channel))
    _y = cp.Variable((n_channel))

    # _x = cp.Parameter((n_channel,))
    # _y = cp.Variable((n_channel,))
    obj = cp.Minimize(cp.sum_squares(_y - _x))
    # cons = [_y >= 0, _y <= amplitude_constraint, cp.sum(_y, axis=0) <= mean_constraint]
    cons = [_y >= 0, _y <= amplitude_constraint]
    prob = cp.Problem(obj, cons)
    layer = CvxpyLayer(prob, parameters=[_x], variables=[_y])
    # x_constrained, = layer(x_example)
    x_constrained, = layer(x_tx)
    # two problem here: 1. how to describe None as the dimension of parameter
    # 2. AttributeError: 'Tensor' object has no attribute 'numpy'
    # ###########################################################
    # Channel part
    # ###########################################################
    SNR_train_dB_train_input = Input(shape=(1,))
    SNR_train = 10. ** (SNR_train_dB_train_input / 10.)  # coverted 7 db of EbNo
    # print ("SNR_train:",SNR_train )
    print(x_constrained)
    print(h)
    # matrix multiply
    # rec_without_noise = tf.matmul(encoded_constraint,
    #                               h,
    #                               transpose_a=False,
    #                               transpose_b=True,
    #                               adjoint_a=False,
    #                               adjoint_b=False,
    #                               a_is_sparse=False,
    #                               b_is_sparse=False,
    #                               name=None)
    [x1_out, x2_out, x3_out] = tf.split(x_constrained, n_channel, 1)
    rec = Lambda(lambda x: (h1 * x[0] + h2 * x[1] + h3 * x[2]),name='MISO_channel')([x1_out, x2_out, x3_out])
    # self-defined normalization
    # rec_without_noise_normalize = (rec_without_noise ) / tf.sqrt(
    #             tf.reduce_mean(
    #                 (rec_without_noise - tf.reduce_mean(rec_without_noise)) ** 2
    #             )
    #         )
    rec_power = tf.reduce_mean( (rec - tf.reduce_mean(rec)) ** 2 )


    print ("rec_without_noise_normalize:",rec[1,:] )
    rec_with_noise = GaussianNoise(
        tf.sqrt(rec_power / SNR_train))(rec)
    # print ("rec_with_noise:",rec_with_noise[1,:] )
    # np.sqrt(1 / SNR_train)

    # ###########################################################
    # RX part
    # ###########################################################
    decoded_s1 = Dense(M, activation='linear',name='hidden_layer_2')(rec_with_noise)
    # decoded_s1 = Dense(M, activation='linear')(decoded_s1)
    # decoded_s1 = Dense(M, activation='linear')(decoded_s1)
    # decoded_s1 = Dense(M, activation='relu')(decoded_s1)
    # decoded_s1 = BatchNormalization(momentum=0, center=False, scale=False)(decoded_s1)
    # decoded1_s1 = Dense(M, activation='relu', name='s1_softmax')(decoded_s1)
    decoded1_s1 = Dense(M, activation='softmax', name='bler_loss')(decoded_s1)

    # ###########################################################
    # AE config
    # ###########################################################
    autoencoder = Model(inputs=[input_signal,SNR_train_dB_train_input], outputs=[decoded1_s1])
    adam = Adam()  # SGD converge much slower than Adam
    # adam = SGD()  # SGD converge much slower than Adam


    # callback
    # class MyCallback(Callback):
    #     def __init__(self, alpha):
    #         self.alpha = alpha
            # self.beta  = beta
            # self.gamma = gamma

        # customize your behavior
        # def on_batch_end(self, epoch, logs={}):
            # results = [logs['bler_loss_loss'], logs['color_loss_loss']]
            # results = [1, 1]
            # K.set_value(self.alpha,  results[0] / (results[0] + results[1]) )
            # print("\n epoch %s, alpha = %s, 1-alpha = %s" % (
            #     epoch + 1, K.get_value(self.alpha), 1-K.get_value(self.alpha)))

        # def on_epoch_end(self, epoch, logs={}):
            # results = [logs['bler_loss_loss'], logs['color_loss_loss']]
            # K.set_value(self.alpha,  results[0] / (results[0] + results[1]) )
            # print("\n epoch %s, alpha = %s" % (
            #     epoch + 1, K.get_value(self.alpha)))

    autoencoder.compile(optimizer=adam,
                        loss={
                            'bler_loss': 'categorical_crossentropy'},
                            # 'color_loss': lambda y_true, y_pred: y_pred},
                        # loss_weights={
                        #     'bler_loss': alpha,
                        #     'color_loss': beta},
                        experimental_run_tf_function=False
                        )
    # printing summary of layers and it's trainable parameters
    print(autoencoder.summary())
    # ###########################################################
    # training auto encoder
    # ###########################################################
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2, mode='min')
    reduce_learn_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1)
    random_const = np.random.randn(N_train, 1)
    # constant = tf.expand_dims(tf.convert_to_tensor(1),axis=0)
    autoencoder.fit([data1,SNR_train_dB_1], [data1],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[reduce_learn_rate
                                ,early_stopping
                                # ,MyCallback(alpha)
                                ]
                    )

    ############################################################################
    # ### Make encoder and decoder
    ###########################################################################
    # making encoder from full autoencoder
    encoder_s1 = Model([input_signal,SNR_train_dB_train_input], encoded_constraint)

    # making decoder from full autoencoder
    dec_input_1 = Input(shape=(1,))
    # deco_s1 = autoencoder.layers[-14](dec_input_1)
    # deco_s1 = autoencoder.layers[-12](deco_s1)
    # deco_s1 = autoencoder.layers[-10](deco_s1)
    # deco_s1 = autoencoder.layers[-9](dec_input_1)
    # deco_s1 = autoencoder.layers[-6](dec_input_1)
    deco_s1 = autoencoder.layers[-3](dec_input_1)
    deco_s1 = autoencoder.layers[-1](deco_s1)
    decoder_s1 = Model(dec_input_1, deco_s1)

    ############################################################################
    # ### Visualize transmitted results
    ############################################################################
    print("Visualize transmitted results")
    scatter_point = []
    for i in range(0, M):
        temp = np.zeros(M)
        temp[i] = 1
        # print(temp)
        scatter_point.append(encoder_s1.predict([np.expand_dims(temp, axis=0),SNR_train_dB_1]))
    scatter_point = np.array(scatter_point)
    scatter_point = scatter_point/np.max(scatter_point) # normalize to 1
    print(scatter_point.shape)
    print(scatter_point)
    print ("average current:",np.mean(scatter_point, axis=0)  )

    # minimum Euclidian distance
    min_Euclidian_distance = 1000.0
    for i in range(0, M):
        for j in range(i+1,M):
            Euclidian_distance_temp = tf.reduce_mean(tf.square( scatter_point[i,:,:] - scatter_point[j,:,:] ))
            if Euclidian_distance_temp < min_Euclidian_distance:
                min_Euclidian_distance = Euclidian_distance_temp
    print ("min_Euclidian_distance:",min_Euclidian_distance)
    # Visualize transmitted results use 3D figure
    fig = plt.figure()
    ax = Axes3D(fig)
    xdata = scatter_point[:,0,0]
    ydata = scatter_point[:,0,1]
    zdata = scatter_point[:,0,2]
    ax.scatter3D(xdata, ydata, zdata, s=100)
    plt.xlabel('Red',color='r')
    plt.ylabel('Green',color='g')
    ax.set_zlabel('Blue',color='b')
    for i_point in range(0,M):
        label = '(%.2f, %.2f, %.2f)' % (scatter_point[i_point,0,0],scatter_point[i_point,0,1],scatter_point[i_point,0,2])
        ax.text(scatter_point[i_point,0,0],scatter_point[i_point,0,1],scatter_point[i_point,0,2],label)
    plt.show()


    ############################################################################
    # Autoencoder BLER(block error rate) performance
    ###########################################################################
    SNR_range_dB = np.arange(0, 21, 2)
    print("SNR_range_dB:", SNR_range_dB)

    bler_s1_sum = np.zeros(len(SNR_range_dB))
    for i_MonteCarlo in range(0, N_MonteCarlo):
        # ------------ generate one hot encoded vector for test-----------------
        test_label_s1 = np.random.randint(M, size=N_test)
        test_data1 = tf.one_hot(test_label_s1, M, on_value=1.0, off_value=0.0)
        nosie_normalize = np.random.randn(N_test, 1)
        # --------------------------------------------------------
        bler_s1 = [None] * len(SNR_range_dB)
        for n in range(0, len(SNR_range_dB)):
            SNR = 10.0 ** (SNR_range_dB[n] / 10.0)
            noise_std = np.sqrt(1 / SNR)
            noise_mean = 0
            noise = noise_std * nosie_normalize  # * 0
            tx_out = encoder_s1.predict([test_data1,SNR_train_dB_1])
            tx_out = tx_out/tf.reduce_max(tx_out)
            [tx1_out, tx2_out, tx3_out] = tf.split(tx_out, n_channel, 1)
            test_r1_without_noise = Lambda(lambda x: (h1 * x[0] + h2 * x[1] + h3 * x[2]))([tx1_out, tx2_out, tx3_out])
            # normalize
            test_r1_without_noise = (test_r1_without_noise ) / tf.sqrt(
                tf.reduce_mean(
                    (test_r1_without_noise - tf.reduce_mean(test_r1_without_noise)) ** 2
                )
            )
            pred_final_signal_s1 = decoder_s1.predict(test_r1_without_noise + noise)
            pred_output_s1 = tf.argmax(pred_final_signal_s1, axis=1)
            bler_s1[n] = sum(1. * (tf.cast(
                pred_output_s1 != tf.convert_to_tensor(test_label_s1, dtype=tf.int64), tf.float32
            ))) / N_test
        print("bler_s1:", bler_s1)
        bler_s1_sum = bler_s1_sum + bler_s1
    bler_s1_mean = bler_s1_sum / N_MonteCarlo
    # record the bler and eclipse constraints
    if(i_color == 0):
        bler_6500K = bler_s1_mean
        print("bler_6500K:", bler_6500K)
        print("\n")
    elif (i_color == 1):
        bler_5700K = bler_s1_mean

############################################################################
# performance visualization
###########################################################################
print("bler_6500K:", bler_6500K)

print("\n")


# #### plot BLER curve
fig = plt.figure()
plt.plot(SNR_range_dB, bler_6500K, 'ro-',
         label='Autoencoder(' + str(n_channel) + ',' + str(k) + ')_s1')

plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid(True)
plt.legend(loc='upper right', ncol=1)
plt.xticks(SNR_range_dB, SNR_range_dB[::1])
plt.show()
print(mpl.get_backend())
