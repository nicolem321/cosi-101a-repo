#!usr/bin/python3.9

# main.py
# Version 1.0.0
# 05-08-22

# Written By: Mason Ware & Novia Wu

''' This file serves as the client and driver for interacting with the CNN. The two classes
    house the client and the CNN itself, respectively. It is worth noting that the CNN class
    in this file is deprecate and only there to serve as a "footprint" so-to-speak. The
    actual network that is used is imported from torchvision.modls (Resnet18)'''


from pathlib import Path
import os
import csv
import argparse

from utils.animate import Loader                                    

from PIL import Image                                               # type: ignore
from tqdm import tqdm                                               # type: ignore
from torch.utils.data import DataLoader                             # type: ignore
from torchvision import datasets, transforms                        # type: ignore
import torch                                                        # type: ignore
import torch.nn as nn                                               # type: ignore
import torch.optim as optim                                         # type: ignore
import torch.nn.functional as F                                     # type: ignore
import matplotlib.pyplot as plt                                     # type: ignore
import torchvision.models as models                                 # type: ignore


class Client:
    ''' A client to interact (train) with the cnn. '''
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.batch_size_train = 64
        self.batch_size_val = 1000
        self.n_epochs = 7
        self.learning_rate = 0.0014
        self.momentum = 0.5
    
    # load the data from the folders
    def get_data(self):
        transform1 = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        transform2 = transforms.Compose([transforms.Resize((224,224)),transforms.GaussianBlur(kernel_size=(5, 9)),transforms.RandomRotation(degrees=(0, 30)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        # ImageFolder automatically assign labels to imgs using the name of their folder
        train_set = []
        for i in range(1,3):
            if i == 1:
                transform = transform1
            else: 
                transform = transform2
            for data in datasets.ImageFolder(self.dir + '/train',transform=transform):
                train_set.append(data)
        val_set = datasets.ImageFolder(self.dir + '/val',transform=transform)   
        img, label = train_set[0]
        print("my input data size: ", img.shape, len(train_set))
        train_loader = DataLoader(train_set, batch_size=self.batch_size_train, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size_val, shuffle=True)
        return train_loader, val_loader
    
    # visualize first 5 images
    def train_imshow(self, train_loader):
        classes = ('1', '10', '2', '3', '4', '5', '6', '7', '8', '9') # Defining the classes we have
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        fig, axes = plt.subplots(figsize=(20, 8), ncols=5)
        for i in range(5):
            ax = axes[i]
            ax.imshow(images[i].permute(1,2,0).squeeze()) 
            ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
        plt.show()
        
    def test(self, model, test_loader, device, verbosity):
        # evaluation, freeze 
        model.eval()
        total_num = 0
        total_correct = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)     
                predict_one_hot = model(data)     
                _, predict_label = torch.max(predict_one_hot, 1)
                if verbosity==1:
                    print("llllll",predict_label)
                total_correct += (predict_label == target).sum().item()
                total_num += target.size(0) 
        return (total_correct / total_num)

    # define the training procedure
    def train(self, model, train_loader, test_loader, device, verbosity, num_epoch='', learning_rate='', momentum=''):
        train_losses = []
        if not num_epoch:
            num_epoch=self.n_epochs
        if not learning_rate:
            learning_rate=self.learning_rate
        if not momentum:
            momentum=self.momentum
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        for epoch in tqdm(range(num_epoch)):
            # train the model
            model.train()
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = network(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
            accuracy = self.test(model, test_loader, device, verbosity)
            if verbosity==1:
                print('accuracy', accuracy)
                print("loss: ",loss)
        
        
# define the cnn model
class CNN(nn.Module):
    ''' A mutlilayer CNN defined using PyTorch. '''
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(5780, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 5780)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Handwriting Recognition CNN")
    parser.add_argument("-r", "--run", action="store_true")
    parser.add_argument("-s", "--save",action="store_true")
    parser.add_argument("-t", "--eval",action="store_true")
    parser.add_argument("--epochs", metavar='             N', type=int)
    parser.add_argument("--learn_rate", metavar='         N.N', type=int)
    parser.add_argument("--dir", metavar='                path/name/', type=str)
    parser.add_argument("--verbosity", metavar='          0 or 1', type=int)
    args = parser.parse_args()
    
    if args.eval:
        client = Client(dir=args.dir[0] if  args.dir else './data/imgs_classified_split')
        train_loader, val_loader = client.get_data()
        device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        network = models.resnet18(pretrained=True)
        client.train(model=network, train_loader=train_loader, 
                    test_loader=val_loader, device=device0,
                    num_epoch=args.epochs[0] if args.epochs else 7,
                    learning_rate=args.learn_rate[0] if args.learn_rate else 0.0014,
                    verbosity=args.verbosity[0] if args.verbosity else 1)
    if args.save:
        # save an eval run as a final model
        if not os.path.exists('final_model.h5'):
            line = '#'*50
            print(f'{line}\nSaving a Copy of CNN Model\n{line}\nepochs = {args.epochs if args.epochs else 7}\n\n')
            loader = Loader("Running Model To Save...", "All done!", 0.05).start()          
            client = Client(dir=args.dir[0] if  args.dir else './data/imgs_classified_split')
            train_loader, val_loader = client.get_data()
            device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            network = models.resnet18(pretrained=True)
            client.train(model=network, train_loader=train_loader, 
                        test_loader=val_loader, device=device0,
                        num_epoch=args.epochs if args.epochs else 7,
                        learning_rate=args.learn_rate[0] if args.learn_rate else 0.0013,
                        verbosity=args.verbosity[0] if args.verbosity else 1)
            torch.save(network.state_dict(), './final_model.h5')
            loader.stop()
        else:
            print(f'\nFinal Model already found, no need to save!\n\n\nEnter the cmd:   python3.9 main.py -r --dir <dir_name>\n\n')
    if args.run:
        network = models.resnet18(pretrained=True)
        network.load_state_dict(torch.load('./final_model.h5'))
        # preprocessing transformation
        transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        if args.dir:
            line = '#'*50
            path = os.getcwd() + (args.dir if not args.dir[0]=='.' else str(args.dir[1:]))
            print(f'{line}\nLoading Model...\n{line}\nUsing dir provided:  {path}\n\n')
            final_results = []
            loader = Loader("Predicting and Writing...", "All done!", 0.05).start()          
            for root, dirs, files in os.walk(args.dir, topdown=False):
                for name in files:
                    cur_path = (os.path.join(root, name))
                    if not cur_path[len(cur_path)-1][0] == '.':
                        img = Image.open(os.getcwd() + '/' + os.path.join(root, name)).convert('RGB')
                        input = transform(img)
                        # unsqueeze batch dimension
                        input = input.unsqueeze(0)
                        network.eval()
                        output = network(input)
                        pred = torch.max(output.data, 1).indices
                        final_results.append({'img':name, 'label':pred[0].item()})
            keys = final_results[0].keys()
            # write output to csv
            with open('./results.csv', 'w', newline=None) as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(final_results)
            loader.stop()
        else:
            line = '#'*50
            print(f'{line}\nNO GIVEN DIRECTORY\n{line}\nPlease provide a directory of images, like so:\n\npython3.9 main.py -r --dir <dir_name>\n\n')

        
        
# matplot lib below (UNUSED)
        
    
# # @timer
# def summarize_diagnostics(histories):
#     ''' plot diagnostic learning curves. '''
#     for i in range(len(histories)):
#         # plot loss
#         plt.subplot(2, 1, 1)
#         plt.title('Cross Entropy Loss')
#         plt.plot(histories[i].history['loss'], color='blue', label='train')
#         plt.plot(histories[i].history['val_loss'], color='orange', label='test')
#         # plot accuracy
#         plt.subplot(2, 1, 2)
#         plt.title('Classification Accuracy')
#         plt.plot(histories[i].history['accuracy'], color='blue', label='train')
#         plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
#     plt.show()
    
# # @timer
# def summarize_performance(scores):
#     ''' summarize model performance. '''
#     # print summary
#     print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
#     # box and whisker plots of results
#     plt.boxplot(scores)
#     plt.show()

# def visualize(client, train_loader) -> None:
#     client.train_imshow(train_loader)
#     for i, (images, labels) in enumerate(train_loader):
#         print(images.shape)
#         break
