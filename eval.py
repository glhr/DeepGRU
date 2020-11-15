
from DeepGRU.predict import model, dataset, predict_single
from DeepGRU.utils.plotter import *
import time

for fold in range(0,3):
    train_loader, test_loader = dataset.get_data_loaders(fold,
                                                         shuffle=True,
                                                         random_seed=1570254494+fold,
                                                         normalize=True,
                                                         batch_size=1)

    LABELS = {1: "handover",
              3: "working",
              0: "distracted",
              2: "waving"}

    LABELS_INV = dict((v, k) for k, v in LABELS.items())

    cm = []
    for c in range(4):
        cm.append([0] * 4)
    for batch in test_loader:
        prediction = predict_single(batch, model, eval=True)
        cm[LABELS_INV[prediction['predicted']]][LABELS_INV[prediction['expected']]] += 1

    plot_confusion_matrix(cm, classes=LABELS_INV, filename=fold)
