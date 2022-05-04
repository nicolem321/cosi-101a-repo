import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm

# Hyper parameters
batch_size_train = 64
batch_size_val = 1000
n_epochs = 80
learning_rate = 0.0013
momentum = 0.5

# load the data from the folders
def get_data():
    data_dir = '/Users/masonware/Desktop/brandeis_cosi/COSI_101A/termProject/data' #change here the path 

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((40,114)),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

    # ImageFolder automatically assign labels to imgs using the name of their folder
    train_set = datasets.ImageFolder(data_dir + '/train',transform=transform)
    val_set = datasets.ImageFolder(data_dir + '/val',transform=transform)
    
    img, label = train_set[0]
    print("my input data size: ", img.shape)

    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size_val, shuffle=True)

    return train_loader, val_loader

# visualize first 5 images
def train_imshow(train_loader):
    classes = ('1', '10', '2', '3', '4', '5', '6', '7', '8', '9') # Defining the classes we have
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    fig, axes = plt.subplots(figsize=(20, 8), ncols=5)
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].permute(1,2,0).squeeze()) 
        ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
    plt.show()
    
    
# define the cnn model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


def test(model, test_loader, device):
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
            print("llllll",predict_label)
            total_correct += (predict_label == target).sum().item()
            total_num += target.size(0)
        
    return (total_correct / total_num)


# define the training procedure
def train(model, train_loader, test_loader, num_epoch, learning_rate, momentum, device):
    train_losses = []
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
        accuracy = test(model, test_loader, device)
        print('accuracy', accuracy)
        print("loss: ",loss)


train_loader, val_loader = get_data()
    
train_imshow(train_loader)
for i, (images, labels) in enumerate(train_loader):
    print(images.shape)
    break

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network = Net().to(device0)
train(model=network, train_loader=train_loader, 
      test_loader=val_loader, num_epoch=n_epochs, 
      learning_rate=learning_rate, momentum=momentum, 
      device=device0)
