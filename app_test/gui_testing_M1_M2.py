# import matplotlib
# matplotlib.use('AGG')
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
import PIL.Image
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
import os
import torch.nn.functional as F

mosquito_transforms = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize([300,300]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def load_image(infilename):
    image = PIL.Image.open(infilename)
    image = mosquito_transforms(image)

    return image


if __name__ == "__main__":

    #Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    #filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    #print(filename)

    # Import the required libraries
    from tkinter import *
    from tkinter import ttk
    import tkinter.filedialog as fd

    # Create an instance of tkinter frame or window
    win = Tk()

    # Set the geometry of tkinter frame
    win.geometry("700x350")


    def open_file():
        global filenames
        filenames = fd.askopenfilenames(parent=win, title='Choose a File')
        print(filenames)

    label = Label(win, text="Select the Button to Open the File", font=('Aerial 11'))
    label.pack(pady=30)

    # Add a Button Widget
    ttk.Button(win, text="Select a File", command=open_file).pack()

    win.wait_window()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("data/M2_app_test.csv")
    label = []

    model = torch.load("model/species/Model_1.pt", map_location=device)
    model.eval()

    out = []

    data_path = './App Test Images/crop_test_pad'

    for i in range(df.shape[0]):
        filename = df.loc[i, 'Image_name']
        if filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            inputs = load_image(file_path)
            outputs = model(inputs.unsqueeze(0).to(device))

            # probabilities
            probs = F.softmax(outputs, 1)
            for j in range(len(outputs)):
                p = probs[j].tolist()
                df.loc[i, 'Confidence_value'] = format(max(p), '.4f')

            # the index of the maximum value in the list , is the predicted class
            value, predicted = torch.max(probs, 1)

            out.append(predicted.cpu().detach().numpy())
            label.append(df.loc[i, 'True_label'])
            df.loc[i, 'Predicted_label'] = predicted.cpu().detach().numpy()
            # print(np.around(value.cpu().detach().numpy(), decimals=4))
            # df.loc[i, 'Confidence_value'] = format(max(list_probs), '.4f')

    out = np.concatenate(out)
    print(out)
    print(np.array(label))
    df.to_csv("data/M2_app_test2.csv", index=False)

    species_all = ["An. funestus",
                   "An. gambiae",
                   "An. other",
                   "Culex",
                   "Other"]
    # print classification report
    print(classification_report(np.array(label), out))

    # make confusion matrix
    conf_mat = confusion_matrix(np.array(label), out)
    conf_mat = conf_mat / np.expand_dims(conf_mat.astype(np.float64).sum(axis=1), 1)
    conf_mat = np.round(conf_mat, decimals=2)
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot()
    hm = sn.heatmap(conf_mat, annot=True, ax=ax, cmap="PuBu", fmt='.2', annot_kws={"size": 35 / np.sqrt(len(conf_mat))})
    ax.set_yticklabels(hm.get_yticklabels(), rotation=90)
    ax.set_xlabel('Predicted labels', fontsize=15)
    ax.set_ylabel('True labels', fontsize=15)
    ax.xaxis.set_ticklabels(species_all)
    ax.yaxis.set_ticklabels(species_all)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('M1_test')
    plt.savefig('model/app_test/M2.png')
    # plt.show()

