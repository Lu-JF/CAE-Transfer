# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:12:11 2023

@author: LJF
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import load_model
from Transfer import dataset_create

img_size = 256
labels=['1','2','3','4','5']
class_num = 5

#scae_path=''
pre_ecnn_path='./pre_ecnn.h5'
fine_ecnn_path='./fine_ecnn.h5'

#source_test_path=''
source_labeled_test_path='.//soure//labeled//Test'
target_test_path='.//target//Test'


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,
                          normalize=True):
    #plot the confusion matrix
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
	#plt.savefig('fusion_matrix.png',dpi=350)
    plt.show()


def model_test(model, data_test, batch_size=None):
    if batch_size==None:
        STEP = 1
    else:
        STEP = data_test.n/batch_size
    score = model.evaluate(data_test, steps=STEP, verbose=1)
    return score

def computing_cm(model, data_test):
    label_pre = model.predict(data_test, verbose=1)
    label_pre = label_pre.argmax(axis=1)
    label_true = data_test.classes
    conf_mat = confusion_matrix(y_true=label_true, y_pred=label_pre)
    return conf_mat

if __name__=='__main__':
    #loading models
    #scae = load_model(scae_path)
    pre_ecnn = load_model(pre_ecnn_path)
    fine_ecnn = load_model(fine_ecnn_path)
    #creating datasets
    source_data_test = dataset_create(source_labeled_test_path, mode = 'categorical')
    target_data_test = dataset_create(target_test_path, mode = 'categorical')
    #computing the test accuracy
    pre_ecnn_score = model_test(pre_ecnn, source_data_test)
    fine_ecnn_score = model_test(fine_ecnn, target_data_test)
    #computing the confusion matrix
    pre_ecnn_cm = computing_cm(pre_ecnn, source_data_test)
    fine_ecnn_cm = computing_cm(fine_ecnn, target_data_test)
    #plot the confusion matrix
    plot_confusion_matrix(pre_ecnn_cm, normalize=False,target_names=labels,title='Confusion Matrix (pre_ECNN)')
    plot_confusion_matrix(fine_ecnn_cm, normalize=False,target_names=labels,title='Confusion Matrix (fine_ECNN)')
