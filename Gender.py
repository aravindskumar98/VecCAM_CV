"""
Gender Classification

0-female 1-male
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from collections import Counter


available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(available_gpus)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print ("running on the GPU")
else:
    device = torch.device("cpu")
    print ("running on the cpu")


class VectorCamDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.data = data
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        x = self.data[index]
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y


# Transforms
from torchvision import transforms

invTrans = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomRotation(degrees=(-45, 45), fill=(0,)),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ColorJitter()
])

EPOCHS = 30
gender_all = ["female", "male"]


def train(train_dataloader, test_dataloader, i):
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    max_val = 0

    for epoch in range(EPOCHS + 1):

        correct = 0
        total = 0
        train_ave_loss = 0
        model.train()
        for batch_X, batch_Y in train_dataloader:
            batch_X = batch_X.cuda()
            batch_Y = batch_Y.cuda()

            # zero gradient
            optimizer.zero_grad()
            # pass through
            outputs = model(batch_X)
            # compute loss and back propagate
            loss = criterion(outputs, batch_Y)

            loss.backward()
            # optimize
            optimizer.step()

            train_ave_loss += loss.data.item()
            _, predicted = outputs.max(1)
            total += batch_Y.size(0)
            correct += predicted.eq(batch_Y).sum().item()

        train_loss.append(train_ave_loss / len(train_dataloader))
        train_acc.append(100. * correct / total)
        print(
            f"Epoch: {epoch},Train Loss: {train_ave_loss / len(train_dataloader)} | Train Acc: {100. * correct / total} ({correct}/{total})")

        # ======  Validation ======
        model.eval()
        valid_correct = 0
        valid_total = 0
        valid_ave_loss = 0

        with torch.no_grad():
            for valid_batch_X, valid_batch_Y in test_dataloader:

                valid_batch_X = valid_batch_X.cuda()
                valid_batch_Y = valid_batch_Y.cuda()

                valid_outputs = model(valid_batch_X)
                loss = criterion(valid_outputs, valid_batch_Y)
                valid_ave_loss += loss.data.item()

                _, predicted = valid_outputs.max(1)
                valid_correct += predicted.eq(valid_batch_Y).sum().item()
                valid_total += valid_batch_Y.size(0)

        valid_loss.append(valid_ave_loss / len(test_dataloader))
        valid_acc.append(100. * valid_correct / valid_total)

        print(
            f"Validation Loss: {valid_ave_loss / len(test_dataloader)} | Validation Acc: {100. * valid_correct / valid_total} ({valid_correct}/{valid_total})")

        if (valid_correct / valid_total) > max_val:
            max_val = valid_correct / valid_total
            print("saved")
            torch.save(model, 'model/sex/M5/sex_fold' + str(i) + '.pt')
    return train_loss, valid_loss, train_acc, valid_acc


def test(test_dataloader, model, threshold=0.5):
    correct = 0
    total = 0
    model.eval()
    out = []
    valid = []
    max_prob = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:

            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = F.softmax(model(inputs), dim=1)

            for j in range(len(outputs)):
                p = outputs[j].tolist()
                if max(p) < threshold:
                    valid.append(False)
                else:
                    valid.append(True)

            max_value, predicted = outputs.max(1)
            max_prob.append(max_value.cpu().detach().numpy())
            out.append(predicted.cpu().detach().numpy())
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print("Accuracy:", round(correct / total, 3))
    return out, max_prob, np.array(valid)


def visualize(true_label, predicted_label, fold=None):
    # print classification report
    print(classification_report(true_label, predicted_label))

    # make confusion matrix
    conf_mat = confusion_matrix(true_label, predicted_label)
    conf_mat = conf_mat / np.expand_dims(conf_mat.astype(np.float64).sum(axis=1), 1)
    conf_mat = np.round(conf_mat, decimals=2)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    # hm = sn.heatmap(conf_mat, annot=True, ax=ax, cmap="PuBu", fmt='d')
    hm = sn.heatmap(conf_mat, annot=True, ax=ax, cmap="PuBu", fmt='.2', annot_kws={"size": 35 / np.sqrt(len(conf_mat))})
    ax.set_yticklabels(hm.get_yticklabels(), rotation=90)
    ax.set_xlabel('Predicted labels', fontsize=15)
    ax.set_ylabel('True labels', fontsize=15)
    ax.xaxis.set_ticklabels(gender_all)
    ax.yaxis.set_ticklabels(gender_all)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # if fold:
    #     plt.title('sex_test_0.9_fold' + str(fold))
    #plt.title('sex_CV_result_0.9')
    plt.savefig(f'model/sex/M5/sex_test_0.8_fold_{fold}.png')
    # plt.show()
    # plt.savefig('model/sex/M1_M2_finetune/sex_test_0.9_fold' + str(fold) + '.png')


iftrain = True
majority = False

true_all = []
predicted_all = []
if iftrain:
    for i in range(1, 6):
        # initialize the model
        # model = models.efficientnet_b1(pretrained=True)

        model = torch.load('model/sex/M1_M2/sex_fold2.pt')
        # print(model)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 2),
        )
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

        # train
        print("In loop")

        # M1_train_data_path = f"data/sex/CV_1_M1/train_data_fold{i}.pt"
        # M1_train_label_path = f"data/sex/CV_1_M1/train_label_fold{i}.pt"
        # M2_train_data_path = f"data/sex/CV_1_M2/train_data_fold{i}.pt"
        # M2_train_label_path = f"data/sex/CV_1_M2/train_label_fold{i}.npy"
        # M3_train_data_path = f"data/gender/CV_1_M3/train_data_fold{i}.pt"
        # M3_train_label_path = f"data/gender/CV_1_M3/train_label_fold{i}.npy"
        M5_train_data_path = f"data/gender/CV_1_M5/train_data_fold{i}.pt"
        M5_train_label_path = f"data/gender/CV_1_M5/train_label_fold{i}.pt"

        # M1_train_data = torch.load(M1_train_data_path)
        # M1_train_label = torch.load(M1_train_label_path)
        # M2_train_data = torch.load(M2_train_data_path)
        # M2_train_label = np.load(M2_train_label_path, allow_pickle=True)
        # M2_train_label = M2_train_label.item()
        # M2_train_label = torch.tensor(M2_train_label["gender"])
        # M3_train_data = torch.load(M3_train_data_path)
        # M3_train_label = np.load(M3_train_label_path, allow_pickle=True)
        # M3_train_label = M3_train_label.item()
        # M3_train_label = torch.tensor(M3_train_label["gender"])
        train_data = torch.load(M5_train_data_path)
        train_label = torch.load(M5_train_label_path)

        # val

        # M1_val_data_path = f"data/sex/CV_1_M1/test_data_fold{i}.pt"
        # M1_val_label_path = f"data/sex/CV_1_M1/test_label_fold{i}.pt"
        # M2_val_data_path = f"data/sex/CV_1_M2/test_data_fold{i}.pt"
        # M2_val_label_path = f"data/sex/CV_1_M2/test_label_fold{i}.npy"
        # M3_val_data_path = f"data/gender/CV_1_M3/val_data_fold{i}.pt"
        # M3_val_label_path = f"data/gender/CV_1_M3/val_label_fold{i}.npy"
        M5_val_data_path = f"data/gender/CV_1_M5/test_data_fold{i}.pt"
        M5_val_label_path = f"data/gender/CV_1_M5/test_label_fold{i}.pt"

        # M1_val_data = torch.load(M1_val_data_path)
        # M1_val_label = torch.load(M1_val_label_path)
        # M2_val_data = torch.load(M2_val_data_path)
        # M2_val_label = np.load(M2_val_label_path, allow_pickle=True)
        # M2_val_label = M2_val_label.item()
        # M2_val_label = torch.tensor(M2_val_label["gender"])
        # M3_val_data = torch.load(M3_val_data_path)
        # M3_val_label = np.load(M3_val_label_path, allow_pickle=True)
        # M3_val_label = M3_val_label.item()
        # M3_val_label = torch.tensor(M3_val_label["gender"])
        test_data = torch.load(M5_val_data_path)
        test_label = torch.load(M5_val_label_path)

        # plt.imshow(torch.permute(M2_val_data[0], (1, 2, 0)))
        # plt.show()

        # Concatenate 2 sets - M1, M2

        # train_data = torch.cat((M1_train_data, M2_train_data), 0)
        # test_data = torch.cat((M1_val_data, M2_val_data), 0)
        #
        # train_label = torch.cat((M1_train_label, M2_train_label), 0)
        # test_label = torch.cat((M1_val_label, M2_val_label), 0)

        print('Train data size: ', train_data.shape)
        print('Train label size: ', train_label.shape)
        print('Val data size: ', test_data.shape)
        print('Val label size: ', test_label.shape)

        train_dataset = VectorCamDataset(train_data, train_label, transform=invTrans)
        test_dataset = VectorCamDataset(test_data, test_label)

        target_list = torch.tensor(train_dataset.labels)
        class_count = np.array([len(np.where(train_dataset.labels == t)[0]) for t in np.unique(train_dataset.labels)])
        print(class_count)
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        print(class_weights)
        class_weights_all = class_weights[target_list]
        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

        train_dataloader = DataLoader(train_dataset, sampler=weighted_sampler, batch_size=32)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # train_loss, valid_loss, train_acc, valid_acc = train(train_dataloader, test_dataloader, i)

        test_model = torch.load('model/sex/M5/sex_fold' + str(i) + '.pt')

        # out, _, _ = test(test_dataloader, test_model)
        #
        # out = np.concatenate(out)

        # misclassified_index = [i for i in range(len(out)) if out[i] != test_label[i]]
        # print(misclassified_index)
        #
        # count = 0

        # for item in misclassified_index:
        #     if count < 20:
        #         plt.imshow(torch.permute(test_dataset[item][0], (1, 2, 0)))
        #         plt.show()
        #         count += 1

        out, _, valid = test(test_dataloader, test_model, threshold=0.8)
        out = np.concatenate(out)
        # misclassified_index = [i for i in range(len(out)) if out[i] != test_label[i]]
        # for idx in misclassified_index:
        #     plt.title(f"true label {gender_all[int(test_dataset[idx][1])]} predicted label {gender_all[out[idx]]}")
        #     plt.imshow(torch.permute(test_dataset[idx][0], (1, 2, 0)))
        #     plt.show()
        omit_count = np.size(valid) - np.count_nonzero(valid)
        #
        print("removed count", omit_count)
        #
        out = out[valid]
        test_label = np.array(test_label)[valid]

        visualize(test_label, out, fold=i)

        np.save(f"model/sex/M5/test_predicted_label_fold{i}.npy", out)
        np.save(f"model/sex/M5/test_true_label_fold{i}.npy", test_label)

        # visualize(test_label, out, fold=i)

        # out = np.load(f"model/sex/M1_M2/test_predicted_label_fold{i}.npy")
        # true = np.load(f"model/sex/M1_M2/test_true_label_fold{i}.npy")
        # true_all.extend(list(test_label))
        # predicted_all.extend(list(out))
else:
    # test
    M1_test_data_path = f"data/gender/CV_1_M1/test_data.pt"
    M1_test_label_path = f"data/gender/CV_1_M1/test_label.pt"
    M2_test_data_path = f"data/gender/CV_1_M2/test_data.pt"
    M2_test_label_path = f"data/gender/CV_1_M2/test_label.npy"
    # M3_test_data_path = f"data/gender/CV_1_M3/test_data.pt"
    # M3_test_label_path = f"data/gender/CV_1_M3/test_label.npy"

    M1_test_data = torch.load(M1_test_data_path)
    M1_test_label = torch.load(M1_test_label_path)
    M2_test_data = torch.load(M2_test_data_path)
    M2_test_label = np.load(M2_test_label_path, allow_pickle=True)
    M2_test_label = M2_test_label.item()
    M2_test_label = torch.tensor(M2_test_label["gender"])
    # M3_test_data = torch.load(M3_test_data_path)
    # M3_test_label = np.load(M3_test_label_path, allow_pickle=True)
    # M3_test_label = M3_test_label.item()
    # M3_test_label = torch.tensor(M3_test_label["gender"])

    test_data = torch.cat((M1_test_data, M2_test_data), 0)
    test_label = torch.cat((M1_test_label, M2_test_label), 0)

    test_dataset = VectorCamDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if majority:
        test_model1 = torch.load('model/gender/M1_M2_M3/gender_fold1.pt')
        test_model2 = torch.load('model/gender/M1_M2_M3/gender_fold2.pt')
        test_model3 = torch.load('model/gender/M1_M2_M3/gender_fold3.pt')
        test_model4 = torch.load('model/gender/M1_M2_M3/gender_fold4.pt')
        test_model5 = torch.load('model/gender/M1_M2_M3/gender_fold5.pt')

        out1, max_prob1, _ = test(test_dataloader, test_model1)
        out1 = np.concatenate(out1)
        max_prob1 = np.concatenate(max_prob1)
        out2, max_prob2, _ = test(test_dataloader, test_model2)
        out2 = np.concatenate(out2)
        max_prob2 = np.concatenate(max_prob2)
        out3, max_prob3, _ = test(test_dataloader, test_model3)
        out3 = np.concatenate(out3)
        max_prob3 = np.concatenate(max_prob3)
        out4, max_prob4, _ = test(test_dataloader, test_model4)
        out4 = np.concatenate(out4)
        max_prob4 = np.concatenate(max_prob4)
        out5, max_prob5, _ = test(test_dataloader, test_model5)
        out5 = np.concatenate(out5)
        max_prob5 = np.concatenate(max_prob5)

        final_result = np.full(len(out1), 7)

        for i in range(len(final_result)):
            results = []
            if max_prob1[i] > 0.8:
                results.append(out1[i])
            if max_prob2[i] > 0.8:
                results.append(out2[i])
            if max_prob3[i] > 0.8:
                results.append(out3[i])
            if max_prob4[i] > 0.8:
                results.append(out4[i])
            if max_prob5[i] > 0.8:
                results.append(out5[i])
            # results = [out1[i], out2[i], out3[i], out4[i], out5[i]]
            c = Counter(results)
            value, count = c.most_common()[0]
            final_result[i] = value

        visualize(np.array(test_label), final_result)

    else:
        test_model = torch.load('model/gender/M1_M2/gender_fold4.pt')
        out, _, valid = test(test_dataloader, test_model, threshold=0.9)
        out = np.concatenate(out)

        omit_count = np.size(valid) - np.count_nonzero(valid)

        print("removed count", omit_count)

        out = out[valid]
        test_label = np.array(test_label)[valid]

        visualize(test_label, out)

# true1 = np.load(f"model/gender/val_true_label_fold1.npy")
# predict1 = np.load(f"model/gender/val_predicted_label_fold1.npy")
# true2 = np.load(f"model/gender/val_true_label_fold2.npy")
# predict2 = np.load(f"model/gender/val_predicted_label_fold2.npy")
# true3 = np.load(f"model/gender/val_true_label_fold3.npy")
# predict3 = np.load(f"model/gender/val_predicted_label_fold3.npy")
# true4 = np.load(f"model/gender/val_true_label_fold4.npy")
# predict4 = np.load(f"model/gender/val_predicted_label_fold4.npy")
# true5 = np.load(f"model/gender/val_true_label_fold5.npy")
# predict5 = np.load(f"model/gender/val_predicted_label_fold5.npy")
#
# true = np.concatenate([true1, true2, true3, true4, true5])
# predict = np.concatenate([predict1, predict2, predict3, predict4, predict5])
# visualize(true, predict)

# visualize(true_all, predicted_all)