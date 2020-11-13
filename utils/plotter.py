import matplotlib.pyplot as plt
from DeepGRU.utils.utils import get_path_from_root
from pathlib import Path
import numpy as np

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
    plt.ylim(0, 5)
    plt.savefig(save_path / '- {}_loss_curve.png'.format(filename))
    plt.show()
    plt.clf()
    plt.close()

    plt.figure(2)
    # plot train/val accuracy
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100)
    plt.savefig(save_path / '- {}_accuracy_curve.png'.format(filename))
    plt.show()
    plt.clf()
    plt.close()

    history = []
