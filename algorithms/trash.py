from utils import *

import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.despine()


data_original = pd.read_csv('/home/poom/Desktop/ML_SET_Demo/data/ORI.BK.csv')[::-1]

WINDOW = 2
EMB_SIZE = 5
STEP = 1
FORECAST = 1

#Create features from previous days data
X_temp = data_original.copy()
X_temp.drop(['Date','Adj Close'], axis=1, inplace=True)
X = X_temp.copy()
for j in range(WINDOW):
    for i in range(len(X_temp.columns)):
        X[X_temp.columns[i]+"-"+str(j+1)] = X_temp.iloc[:,i].shift((j+1)*(-1))
X['Future Close'] = X_temp['Close'].shift(FORECAST)

#Create labels
X['Up'] = X['Future Close'] > X['Close']
X['Down'] = X['Future Close'] <= X['Close']
X = X.dropna()
y_reg = X['Future Close']
y_class = X[['Up','Down']]
X.drop(['Future Close', 'Up', 'Down'], axis=1, inplace=True)

#split into train, test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y_class, test_size=0.3, random_state=101)

### set all variables
# number of neurons in each layer
input_num_units = len(X.columns)
hidden_num_units = 10
output_num_units = len(y_class.columns)
# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])
# set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01
### define weights and biases of the neural network
weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units])),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units])),
    'output': tf.Variable(tf.random_normal([output_num_units],))
}
#create nn computational graph
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)
output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
#define cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
#set optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#initialize variables
init = tf.initialize_all_variables()

#create and run session
with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
            
        print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
    
    print "\nTraining complete!"
    
    
    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)})
    
    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: test_x.reshape(-1, input_num_units)})
#model = Sequential()
#model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE),
#                        nb_filter=16,
#                        filter_length=4,
#                        border_mode='same'))
#model.add(BatchNormalization())
#model.add(LeakyReLU())
#model.add(Dropout(0.5))
#
#model.add(Convolution1D(nb_filter=8,
#                        filter_length=4,
#                        border_mode='same'))
#model.add(BatchNormalization())
#model.add(LeakyReLU())
#model.add(Dropout(0.5))
#
#model.add(Flatten())
#
#model.add(Dense(64))
#model.add(BatchNormalization())
#model.add(LeakyReLU())
#
#
#model.add(Dense(2))
#model.add(Activation('softmax'))
#
#opt = Nadam(lr=0.002)
#
#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
#checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)
#
#
#model.compile(optimizer=opt, 
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#
#history = model.fit(X_train, Y_train, 
#          nb_epoch = 100, 
#          batch_size = 128, 
#          verbose=1, 
#          validation_data=(X_test, Y_test),
#          callbacks=[reduce_lr, checkpointer],
#          shuffle=True)
#
#model.load_weights("lolkek.hdf5")
#pred = model.predict(np.array(X_test))
#
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])
#
#print C / C.astype(np.float).sum(axis=1)
#
## Classification
## [[ 0.75510204  0.24489796]
##  [ 0.46938776  0.53061224]]
#
#
## for i in range(len(pred)):
##     print Y_test[i], pred[i]
#
#
#plt.figure()
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='best')
#plt.show()
#
#plt.figure()
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='best')
#plt.show()
#
