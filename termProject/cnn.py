import argparse
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm

class Client:
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.batch_size_train = 64
        self.batch_size_val = 1000
        self.n_epochs = 80
        self.learning_rate = 0.0013
        self.momentum = 0.5
    
    # load the data from the folders
    def get_data(self):
        data_dir = '/Users/masonware/Desktop/brandeis_cosi/COSI_101A/termProject/imgs_classified_split' #change here the path 
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((40,114)),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

        # ImageFolder automatically assign labels to imgs using the name of their folder
        train_set = datasets.ImageFolder(data_dir + '/train',transform=transform)
        val_set = datasets.ImageFolder(data_dir + '/val',transform=transform)
        
        img, label = train_set[0]
        print("my input data size: ", img.shape)

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
        # 1, define optimizer
        # "TODO: try different optimizer"
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)

        for epoch in tqdm(range(num_epoch)):
            # train the model
            model.train()
            for i, (data, target) in enumerate(train_loader):
                
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                
                # 2, forward
                output = network(data)
                
                # 3, calculate the loss
                "TODO: try use cross entropy loss instead "
                loss = F.nll_loss(output, target)
                
                # 4, backward
                loss.backward()
                optimizer.step()
            # evaluate the accuracy on test data for each epoch
            accuracy = self.test(model, test_loader, device, verbosity)
            if verbosity==1:
                print('accuracy', accuracy)
                print("loss: ",loss)
        
        
# define the cnn model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(3500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20*7*25)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)

    
    
    
# Hyper parameters
# batch_size_train = 64
# batch_size_val = 1000
# n_epochs = 80
# learning_rate = 0.0013
# momentum = 0.5
def visualize(client, train_loader) -> None:
    client.train_imshow(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Handwriting Recognition CNN")
    parser.add_argument("--run", "-r", action="store_true")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--eval", "-t", action="store_true")
    parser.add_argument("--epochs", metavar='N', type=int, nargs='+')
    parser.add_argument("--learn_rate", metavar='N', type=int, nargs='+')
    parser.add_argument("--dir", "-d", metavar='N', type=int, nargs='+')
    parser.add_argument("--verbosity", metavar='N', type=int, nargs='+')
    args = parser.parse_args()
    
    if args.eval:
        client = Client(dir=args.dir[0] if  args.dir else './imgs_classified_split')
        train_loader, val_loader = client.get_data()
        device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        network = CNN().to(device0)
        client.train(model=network, train_loader=train_loader, 
                    test_loader=val_loader, device=device0,
                    num_epoch=args.epochs[0] if args.epochs else 80,
                    learning_rate=args.learn_rate[0] if args.learn_rate else 0.0013,
                    verbosity=args.verbosity[0] if args.verbosity else 1)
    if args.save:
        # save an eval run as a final model
        if not os.path.exists('final_model.h5'):
            line = '#'*50
            print(f'{line}\nSaving a Copy of CNN Model\n{line}\nepochs = {args.epochs[0] if args.epochs else 80}\n\n')
            client = Client(dir=args.dir[0] if  args.dir else './imgs_classified_split')
            train_loader, val_loader = client.get_data()
            device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            network = CNN().to(device0)
            client.train(model=network, train_loader=train_loader, 
                        test_loader=val_loader, device=device0,
                        num_epoch=args.epochs[0] if args.epochs else 80,
                        learning_rate=args.learn_rate[0] if args.learn_rate else 0.0013,
                        verbosity=args.verbosity[0] if args.verbosity else 1)
            torch.save(network.state_dict(), './final_model.h5')    # ?
        else:
            print(f'\nFinal Model already found, no need to save!\n\n\nEnter the cmd:   python main.py --run --k N')
    if args.run:
        pass
        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()
    
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
