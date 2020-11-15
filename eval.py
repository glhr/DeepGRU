
from DeepGRU.predict import dataset, get_model, predict_single
from DeepGRU.utils.plotter import *
import time

models = {
    0: "save/lh7/3-fold cross-subject CV/LH7-1605455681.0777423-fold0.pt",
    1: "save/lh7/3-fold cross-subject CV/LH7-1605456785.0970922-fold1.pt",
    2: "save/lh7/3-fold cross-subject CV/LH7-1605459149.0506065-fold2.pt",
}

for fold in range(0,3):
    train_loader, test_loader = dataset.get_data_loaders(fold,
                                                         shuffle=True,
                                                         random_seed=1570254494+fold,
                                                         normalize=True,
                                                         batch_size=1)

    model = get_model(path=models[fold])


    cm = []
    for c in range(4):
        cm.append([0] * 4)
    for batch in test_loader:
        prediction = predict_single(batch, model, eval=True)
        cm[dataset.class_to_idx[prediction['predicted']]][dataset.class_to_idx[prediction['expected']]] += 1

    plot_confusion_matrix(cm, classes=dataset.class_to_idx, filename=fold)
