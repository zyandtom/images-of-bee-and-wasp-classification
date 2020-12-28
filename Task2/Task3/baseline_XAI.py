if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import os
    import cv2
    import time
    import torch
    from torchvision import datasets, transforms, models
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.optim as optim
    from torchsummary import summary
    import numpy as np
    import matplotlib.pyplot as plt

    # os.chdir('C:\Code\ML\Project')
    data = pd.read_csv("kaggle_bee_vs_wasp/labels.csv", index_col=False)

    #change the path in the feature = 'path'
    for i in data.index:
        data['path'].iloc[i] = data['path'].iloc[i].replace('\\', '/')
        
    # transform the label to class number
    labeltool = LabelEncoder()
    labeltool.fit(data['label'])
    classes = labeltool.classes_
    data['label'] = labeltool.transform(data['label'])



    # train data
    traindf = data[(data['is_validation'] == 0) & (data['is_final_validation'] == 0)]

    # validation data
    validationdf = data[data['is_validation'] == 1]

    # test data
    testdf = data[data['is_final_validation'] == 1]

    traindf['label'] = traindf['label'].astype(np.int64)
    validationdf['label'] = validationdf['label'].astype(np.int64)
    testdf['label']  = testdf['label'] .astype(np.int64)

    # definition of trransform method
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          #transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406], [0.229,0.225,0.224])])

    valid_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406], [0.229,0.225,0.224])])



    # definition of dataset
    class BeeDataset(Dataset):
        # initiate the components
        def __init__(self, df: pd.DataFrame, imgdir: str, train: bool,
                     transforms=None):

            self.df = df
            self.imgdir = imgdir
            self.train = train
            self.transforms = transforms

        def __getitem__(self, index):
            # join the file path
            im_path = os.path.join(self.imgdir, self.df.iloc[index]["path"])
            x = cv2.imread(im_path)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (224, 224))

            if self.transforms:
                x = self.transforms(x)

            if self.train:
                y = self.df.iloc[index]["label"]
                return x, y
            else:
                return x

        def __len__(self):
            return len(self.df)
        
        def getbatch(self, indices):
            images = []
            labels = []
            for index in indices:
                image, label = self.__getitem__(index)
                images.append(image)
                labels.append(label)
                print(image)
                print(label)
            return torch.stack(images), torch.tensor(labels)


    # define a network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # load the pretrained models
            self.model = models.resnet50(pretrained=False)
            # self.model = models.resnet50_bn(pretrained = True)
            # self.model = models.vgg16(pretrained = True)
            # self.model = models.vgg16_bn(pretrained = True)
            self.model.fc = nn.Linear(2048, 4)

        def forward(self, x):
            output = self.model(x)
            return output
        
        


    # define the dataset
    train_dataset = BeeDataset(df=traindf, imgdir='kaggle_bee_vs_wasp', train=True,
                               transforms=train_transform)
    valid_dataset = BeeDataset(df=validationdf, imgdir='kaggle_bee_vs_wasp', train=True,
                               transforms=valid_transform)
    test_dataset = BeeDataset(df=testdf, imgdir='kaggle_bee_vs_wasp', train=True,
                              transforms=valid_transform)



    # define the dataloader
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=64, num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=8, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=8, num_workers=0)



    # check whether cuda can be used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)



    # define the loss function and the optimization method
    criterion = nn.CrossEntropyLoss()

    net = Net()
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)



    # visualize the network architecture
    #summary(model=net, input_size=(3, 224, 224), batch_size=8)


    def normalize(image):
        return (image - image.min()) / (image.max() - image.min())


    def compute_saliency_maps(x, y, model):
        model.eval()
        x = x.cuda()
        x.requires_grad_()

        y_pred = model(x)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(y_pred, y.cuda())
        loss.backward()

        saliencies = x.grad.abs().detach().cpu()

        saliencies = torch.stack([normalize(item) for item in saliencies])
        return saliencies

    # define the training function
    def train_model(model, optimizer, n_epochs, criterion):
        # start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            epoch_time = time.time()
            epoch_loss = 0
            correct = 0
            total = 0
            print("Epoch {}/{}".format(epoch, n_epochs))

            #########################train the model
            model.train()

            # for inputs,labels in train_loader:
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                # get the inputs and labels in training data
                inputs = inputs.to(device)
                labels = (labels).to(device)
                #print(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # compute training loss
                epoch_loss += loss.item()
                if i % 250 == 249:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch, i + 1, epoch_loss / 250))
                    epoch_loss = 0.0

                # compute training accuracy
                _, pred = torch.max(output, 1)
                correct += (pred.cpu() == labels.cpu()).sum().item()
                total += labels.shape[0]
            acc = correct / total
            #######################################saliency map
            img_indices = [1, 2, 3, 4]
            images, labels = train_dataset.getbatch(img_indices)
            saliencies = compute_saliency_maps(images, labels, net)

            fig, axs = plt.subplots(2, len(img_indices), figsize=(20, 8))
            for row, target in enumerate([images, saliencies]):
                for column, img in enumerate(target):
                    # print(img)
                    axs[row][column].imshow(img.permute(1, 2, 0).numpy())
                    axs[row][column].set_title('label: %s' % classes[column])
            # plt.show()
            plt.savefig("images/test.jpg")
            print('Train accuracy is:{:.4f}'.format(acc))
            #####################################evaluation
        #     model.eval()
        #     a = 0
        #     pred_val = 0
        #     corr = 0
        #     tot = 0
        #
        #     with torch.no_grad():
        #         for val_inp, val_label in valid_loader:
        #             val_inp = val_inp.to(device)
        #             val_label = val_label.to(device)
        #
        #             # forward
        #             out_val = model(val_inp)
        #             loss = criterion(out_val, val_label)
        #
        #             # compute evaluation loss
        #             a += loss.item()
        #
        #             # compute evaluation accuracy
        #             _, pred_val = torch.max(out_val, 1)
        #             corr += (pred_val.cpu() == val_label.cpu()).sum().item()
        #             tot += val_label.shape[0]
        #         acc_val = corr / tot
        #
        #     # print
        #     epoch_time2 = time.time()
        #     print("Duration : {:.4f},Train Loss :{:.4f},Train Acc :{:.4f}, Valid Loss:{:.4f},Valid acc :{:.4f}".format(
        #         epoch_time2 - epoch_time, epoch_loss / len(labels), acc, a / len(val_label), acc_val))
        # end_time = time.time()
        # print("Total time :{:.0f}s".format(end_time - start_time))




    # # define the test function
    # def eval_model(model):
    #     correct = 0
    #     total = 0
    #
    #     model.eval()
    #     with torch.no_grad():
    #         for images, label in test_loader:
    #             images = images.to(device)
    #             label = label.to(device)
    #
    #             # load image to the model
    #             output = model(images)
    #
    #             # compute the test accracy
    #             _, pred = torch.max(output, 1)
    #             correct += (pred == label).sum().item()
    #             total += label.shape[0]
    #     print("The accuracy in Test dataset is %d %%" % (100 * correct / total))


    # Train!
    train_model(model=net, optimizer=optimizer, n_epochs=20, criterion=criterion)
    
#%%%%%%%%%%%%%%%%   saliency map






     
        
     
