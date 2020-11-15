import matplotlib.pyplot as plt
from DeepGRU.utils.utils import get_path_from_root
from pathlib import Path
import numpy as np
import itertools
import time

save_path = Path(get_path_from_root("save/"))

history = []

def add_to_history(avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc):
    global history
    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

def plt_result(filename="SC"):
    global history

    history = np.array(history)

    print(save_path / '- {}_loss_curve.png')
    print(save_path / '- {}_accuracy_curve.png')

    plt.figure(1)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, np.max(history[:, 0:2]))
    plt.savefig(save_path / '{}_loss_curve.png'.format(filename))
    plt.clf()
    plt.close()

    plt.figure(2)
    # plot train/val accuracy
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100)
    plt.savefig(save_path / '{}_accuracy_curve.png'.format(filename))
    plt.clf()
    plt.close()

    history = []




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          filename=time.time()):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.array(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path / "{}_confusion matrix.png".format(filename))
