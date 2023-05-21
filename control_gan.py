import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.utils.data import random_split

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from sklearn.metrics import classification_report


workers = 2
batch_size = 256
image_size = 64
num_channels = 3
noise_dim = 100
ngf = 64
ndf = 64
num_epochs = 50
lr = 0.0002
ngpu = 1

data_dir = '/home/bhagesh20558/code/cv/'
img_dir = data_dir + "dataset/"
attribute_dir = data_dir + 'list_attr_celeba.txt'

plots_save_dir = '/home/bhagesh20558/code/cv/plots/experiment_1'
classifier_plots_dir = plots_save_dir + 'classifier_plots/'
models_save_dir = '/home/bhagesh20558/code/cv/models/experiment_1'
log_dir = models_save_dir + 'log.txt'

try:
    os.mkdir(plots_save_dir)
    os.mkdir(classifier_plots_dir)
    os.mkdir(models_save_dir)

except:
    pass

device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

def get_attributes(attribute_dir, reqd_labels):
    f = open(attribute_dir,'r')
    lines = f.readlines()
    column_names_str = lines[1]
    data = lines[2:]

    column_names = []
    for col_name in column_names_str.split(' '):
        col_name = col_name.strip()
        if (len(col_name)) != 0:
            column_names.append(col_name)

    print(column_names, len(column_names))   

    rows = []
    for line in data:
        # line = line.split(' ')[2:]
        line = line.split(' ')[1:] # for celebA
        # print(line)
        row = []
        for attribute_val in line:
            if (len(attribute_val) != 0):
                attribute_val = int(attribute_val.strip())
                row.append(attribute_val)
        # print(len(row))
        if(len(row) == 41):
            print(row)
        rows.append(row)

    df = pd.DataFrame(rows, columns = column_names)
    correct = {1:1, -1:0}
    for column_name in column_names:
        df[column_name] = df[column_name].apply(lambda x: correct[x])
    df = df[reqd_labels]
    return df

reqd_labels = ['Attractive', 'Bald', 'Chubby', 'Eyeglasses', 'Male', 'Smiling', 'Young']
reqd_labels = [
    'Attractive', 'Bald', 'Black_Hair', 'Blond_Hair', 'Chubby', 'Eyeglasses',  'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Young'
]
attribute_df = get_attributes(attribute_dir, reqd_labels=reqd_labels)
features_considered = len(reqd_labels)



class CelebADataloader(Dataset):
    def __init__(self, img_dir, attribute_df, imgtransform):
        self.img_dir = img_dir
        self.attribute_df = attribute_df.to_numpy()
        self.ds = dset.ImageFolder(
            img_dir, 
            transform = imgtransform
        )
        
    def __getitem__(self, idx):
        return (self.ds[idx][0], self.attribute_df[idx])

    def __len__(self):
        return len(self.ds)



transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

multi_celeb_dataset = CelebADataloader(img_dir, attribute_df, transform)
n = len(multi_celeb_dataset)
train_size = int(0.9 * n)
val_size = n - train_size
train, val = random_split(multi_celeb_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
dl = DataLoader(multi_celeb_dataset, batch_size=batch_size, shuffle=True)
train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(val, batch_size=batch_size, shuffle=True)


for x, y in dl:
    print(x.shape, y.shape)
    break



class Generator(nn.Module):
    def __init__(self, noise_dim = 100, ngf = 64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( noise_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


generator = Generator(noise_dim + features_considered).to(device)
print(generator)

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input (num_channels) x 64 x 64
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

discrim = Discriminator(ndf).to(device)
print(discrim)


class Classifier(nn.Module):

    def __init__(self, im_chan=3, n_classes=7, hidden_dim=64):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            self.get_block(im_chan, hidden_dim),
            self.get_block(hidden_dim, hidden_dim * 2),
            self.get_block(hidden_dim * 2, hidden_dim * 4, stride=3),
            self.get_block(hidden_dim * 4, n_classes, final_layer=True),
        )
        self.sigmoid = nn.Sigmoid()

    def get_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, image):
        class_pred = self.classifier(image)
        class_pred = class_pred.view(len(class_pred), -1)
        class_pred_sig = self.sigmoid(class_pred)
        return class_pred, class_pred_sig


def validation(model, dataloader):
    model.eval()
    label_indices = range(features_considered)
    predictions, class_labels = [], []
    val_losses = []

    criterion = nn.BCEWithLogitsLoss()

    for real, labels in tqdm(dataloader):
        real = real.to(device)
        labels = labels[:, label_indices].to(device).float()

        class_pred, class_pred_sig = model(real)
        class_pred_sig = (class_pred_sig > 0.5).float() 
        class_loss = criterion(class_pred, labels)
        
        val_losses += [class_loss.item()] #

        predictions += [*class_pred_sig.cpu().numpy()]
        class_labels += [*labels.cpu().numpy()]

    report = classification_report(class_labels, predictions, target_names=reqd_labels, zero_division=1, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    return f1_score, sum(val_losses) / len(val_losses)


def train_classifier(filename, dataloader, valid_dl):

    label_indices = range(features_considered)

    n_epochs = 6
    display_step = 100
    lr = 0.001
    beta_1 = 0.5
    beta_2 = 0.999


    classifier = Classifier(n_classes=len(label_indices)).to(device)
    class_opt = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(beta_1, beta_2))
    criterion = nn.BCEWithLogitsLoss()

    cur_step = 0
    classifier_losses, train_losses, val_losses = [], [], []
    predictions, class_labels = [], []
    train_f1_scores, val_f1_scores = [], []

    for epoch in range(n_epochs):
        
        for real, labels in tqdm(dataloader):
            real = real.to(device)
            labels = labels[:, label_indices].to(device).float()

            class_opt.zero_grad()
            class_pred, class_pred_sig = classifier(real)
            class_pred_sig = (class_pred_sig > 0.5).float() 
            class_loss = criterion(class_pred, labels)
            class_loss.backward() # Calculate the gradients
            class_opt.step() # Update the weights
            classifier_losses += [class_loss.item()] # Keep track of the average classifier loss

            predictions += [*class_pred_sig.cpu().numpy()]
            class_labels += [*labels.cpu().numpy()]


        class_mean = sum(classifier_losses[-display_step:]) / display_step
        print(f"Step {cur_step}: Classifier loss: {class_mean}")                

        train_loss = sum(classifier_losses) / len(classifier_losses)
        train_losses.append(train_loss)

        report = classification_report(class_labels, predictions, target_names=reqd_labels, zero_division=1, output_dict=True)
        train_f1_score = report['weighted avg']['f1-score']
        train_f1_scores.append(train_f1_score)

        val_f1_score, val_loss = validation(classifier, valid_dl)
        val_f1_scores.append(val_f1_score)
        val_losses.append(val_loss)
        classifier.train()

        print('Epoch : {}, Train F1 Score : {}, Valid F1 Score : {}'.format(epoch, train_f1_score, val_f1_score))

        cur_step += 1


        if epoch % 5 == 0:
            torch.save(classifier, models_save_dir + 'classifier_{}.pt'.format(epoch))

            ax = plt.subplot()
            epochs = [i for i in range(1, len(train_losses) + 1)]
            train_line = ax.plot(epochs, train_losses, label="Training Loss")
            train_line = ax.plot(epochs, val_losses, label="Validation Loss")
            legend = ax.legend(loc='lower right')
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Train Loss")
            plt.savefig(classifier_plots_dir + 'classifier_loss.png')
            plt.show()
            plt.close()

            ax = plt.subplot()
            epochs = [i for i in range(1, len(train_f1_scores) + 1)]
            train_line = ax.plot(epochs, train_f1_scores, label="Training F1 Score")
            val_line = ax.plot(epochs, val_f1_scores, label="Valid F1 Score")
            legend = ax.legend(loc='lower right')
            ax.set_xlabel("Iterations")
            ax.set_ylabel("F1 Scores")
            plt.savefig(classifier_plots_dir + 'classifier_f1_score.png')
            plt.show()
            plt.close()

    torch.save(classifier, filename)
    return classifier


def train_classifier_full(filename, dl):

    label_indices = range(features_considered)

    n_epochs = 10
    display_step = 100
    lr = 0.00001
    beta_1 = 0.5
    beta_2 = 0.999


    classifier = Classifier(n_classes=len(label_indices)).to(device)
    # print(summary(classifier,(3,64,64)))
    class_opt = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(beta_1, beta_2))
    criterion = nn.BCEWithLogitsLoss()

    cur_step = 0
    classifier_losses, train_losses, val_losses = [], [], []
    predictions, class_labels = [], []
    train_f1_scores, val_f1_scores = [], []

    for epoch in range(n_epochs):
        
        for real, labels in tqdm(dl):
            real = real.to(device)
            labels = labels[:, label_indices].to(device).float()

            class_opt.zero_grad()
            class_pred, class_pred_sig = classifier(real)
            class_pred_sig = (class_pred_sig > 0.5).float() 
            class_loss = criterion(class_pred, labels)
            class_loss.backward() # Calculate the gradients
            class_opt.step() # Update the weights
            classifier_losses += [class_loss.item()] # Keep track of the average classifier loss

            predictions += [*class_pred_sig.cpu().numpy()]
            class_labels += [*labels.cpu().numpy()]


        class_mean = sum(classifier_losses[-display_step:]) / display_step
        print(f"Step {cur_step}: Classifier loss: {class_mean}")                

        train_loss = sum(classifier_losses) / len(classifier_losses)
        train_losses.append(train_loss)

        report = classification_report(class_labels, predictions, target_names=reqd_labels, zero_division=1, output_dict=True)
        train_f1_score = report['weighted avg']['f1-score']
        train_f1_scores.append(train_f1_score)

        print('Epoch : {}, F1 Score : {}'.format(epoch, train_f1_score))
        cur_step += 1

        if epoch % 5 == 0:
            torch.save(classifier, models_save_dir + 'full_classifier_{}.pt'.format(epoch))

            ax = plt.subplot()
            epochs = [i for i in range(1, len(train_losses) + 1)]
            train_line = ax.plot(epochs, train_losses, label="Training Loss")
            legend = ax.legend(loc='lower right')
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Train Loss")
            plt.savefig(classifier_plots_dir + 'full_classifier_loss.png')
            plt.show()
            plt.close()

            ax = plt.subplot()
            epochs = [i for i in range(1, len(train_f1_scores) + 1)]
            train_line = ax.plot(epochs, train_f1_scores, label="Training F1 Score")
            legend = ax.legend(loc='lower right')
            ax.set_xlabel("Iterations")
            ax.set_ylabel("F1 Scores")
            plt.savefig(classifier_plots_dir + 'full_classifier_f1_score.png')
            plt.show()
            plt.close()

    torch.save(classifier, filename)
    return classifier


def load_classifier(filename):
    model = torch.load(filename)
    model.eval()
    return model


classifier_dir = models_save_dir + 'classifier.pt'

classifier = None
if os.path.exists(classifier_dir):
    classifier = load_classifier(classifier_dir).to(device)
else:
    classifier = train_classifier(classifier_dir, train_dl, valid_dl)


# full_classifier_dir = models_save_dir + 'full_classifier.pt'

# full_classifier = None
# if os.path.exists(full_classifier_dir):
#     full_classifier = load_classifier(full_classifier_dir).to(device)
# else:
#     full_classifier = train_classifier_full(full_classifier_dir, dl)


for param in classifier.parameters():
    param.requires_grad = False


criterion = nn.BCELoss()
criterion2 = nn.BCEWithLogitsLoss()

real_label = 1.
fake_label = 0.

#reqd_labels = ['Attractive', 'Bald', 'Chubby', 'Eyeglasses', 'Male', 'Smiling', 'Young']
# reqd_labels = [
#     'Attractive', 'Bald', 'Black_Hair', 'Blond_Hair', 'Chubby', 'Eyeglasses',  'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Young'
# ]
fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)
fixed_attributes = torch.Tensor([1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]).to(device)
fixed_attributes = fixed_attributes.repeat(64, 1).reshape((64, features_considered, 1, 1))
print(fixed_attributes.shape)
fixed_noise = torch.cat((fixed_noise, fixed_attributes), 1)
print(fixed_noise.shape)

optimizerD = optim.Adam(discrim.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))



img_list = []
G_losses = []
D_losses = []
iters = 0
avg_G_losses, avg_D_losses = [], []

for epoch in range(num_epochs+1):
    g_batch_loss, d_batch_loss = 0, 0

    for i, data in tqdm(enumerate(dl, 0)):

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

        discrim.zero_grad()
        image = data[0].to(device)
        attribute_labels = data[1].to(device).float()

        batch_size = image.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        output = discrim(image).view(-1)
        d_real_loss = criterion(output, label)
        d_real_loss.backward()
        D_x = output.mean().item()

        attribute_labels_reshape = torch.reshape(attribute_labels, (attribute_labels.shape[0], attribute_labels.shape[1], 1, 1))
        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        gen_input = torch.cat((noise, attribute_labels_reshape), 1)
        fake = generator(gen_input)
        label.fill_(fake_label)
        output = discrim(fake.detach()).view(-1)
        d_fake_loss = criterion(output, label)
        d_fake_loss.backward()
        D_G_z1 = output.mean().item()
        d_loss = d_real_loss + d_fake_loss
        optimizerD.step()


        # (2) Update G network: maximize log(D(G(z)))

        generator.zero_grad()
        label.fill_(real_label) 
        output = discrim(fake).view(-1)
        g_discrim_loss = criterion(output, label)
        g_discrim_loss.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        
        classifier_out, classifier_out_sigmoid = classifier(fake)
        g_class_loss = criterion2(classifier_out, attribute_labels)
        g_class_loss.backward()
        optimizerG.step()
        g_loss = g_discrim_loss + g_class_loss

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        g_batch_loss += g_loss.item()
        d_batch_loss += d_loss.item()

        if (iters % 2000 == 0) or ((epoch == num_epochs-1) and (i == len(dl)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1,2,0)), animated=True)
            plt.savefig(plots_save_dir + 'check_{}.png'.format(iters))
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_losses, label="G")
            plt.plot(D_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(plots_save_dir + "losses.png")
            plt.show()
            plt.close()

        iters += 1

    g_batch_loss /= len(dl)
    d_batch_loss /= len(dl)
    avg_G_losses.append(g_batch_loss)
    avg_D_losses.append(d_batch_loss)

    if epoch % 5 == 0:
        gen_dir = models_save_dir + 'generator_{}.pt'.format(epoch)
        discrim_dir = models_save_dir + 'discriminator_{}.pt'.format(epoch)
        torch.save(generator, gen_dir)
        torch.save(discrim, discrim_dir)



    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(avg_G_losses, label="G")
    plt.plot(avg_D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plots_save_dir + "losses_avg.png")
    plt.show()
    plt.close()



