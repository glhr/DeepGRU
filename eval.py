
from DeepGRU.predict import dataset, get_model, predict_single
from DeepGRU.utils.plotter import *
import time

models = {
    0: "save/lh7/3-fold cross-subject CV/LH7-1605455681.0777423-fold0.pt",
    1: "save/lh7/3-fold cross-subject CV/LH7-1605456785.0970922-fold1.pt",
    2: "save/lh7/3-fold cross-subject CV/LH7-1605459149.0506065-fold2.pt",
}

models = {
    0: "save/lh7/5-fold random CV/LH7-1605454350.520024-fold0.pt",
    1: "save/lh7/5-fold random CV/LH7-1605454595.2738004-fold1.pt",
    2: "save/lh7/5-fold random CV/LH7-1605454838.9315948-fold2.pt",
    3: "save/lh7/5-fold random CV/LH7-1605455107.765804-fold3.pt",
    4: "save/lh7/5-fold random CV/LH7-1605455339.0769584-fold4.pt",
}

for fold in range(0,len(models)):
    train_loader, test_loader = dataset.get_data_loaders(fold,
                                                         shuffle=True,
                                                         random_seed=1570254494+fold,
                                                         normalize=True,
                                                         batch_size=1)

    model = get_model(path=models[fold])

    cm = []
    num_classes = len(dataset.class_to_idx)
    for c in range(num_classes):
        cm.append([0] * num_classes)
    correct = 0
    print(len(test_loader))
    for batch in test_loader:
        prediction = predict_single(batch, model, eval=True)
        expected = dataset.class_to_idx[prediction['expected']]
        predicted = dataset.class_to_idx[prediction['predicted']]
        cm[expected][predicted] += 1

        if expected == predicted:
            correct += 1
        # cm[0][2] += 1

    print(f"Accuracy: {100*correct/len(test_loader):.2f}")

    plot_confusion_matrix(cm, classes=dataset.class_to_idx, filename=fold)
