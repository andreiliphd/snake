import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog
import webbrowser
import time
# import matplotlib.pyplot as plt
# import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets

## In[2]:
#
#
#transforms_image = transforms.Compose([transforms.Resize(32),
#                                     transforms.CenterCrop(32),
#                                     transforms.ToTensor()])
#train_xray = torch.utils.data.DataLoader(datasets.ImageFolder('chest_xray/train', 
#                                                                            transform=transforms_image),
#                                                        batch_size=20, shuffle=True)
#def imshow(img):
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
#"""
#DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
#"""
## obtain one batch of training images
#dataiter = iter(train_xray)
#images, _ = dataiter.next() # _ for no labels
#
## plot the images in the batch, along with the corresponding labels
#fig = plt.figure(figsize=(20, 4))
#plot_size=20
#for idx in np.arange(plot_size):
#    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
#    imshow(images[idx])
#

# In[3]:



# In[4]:



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(in_features=6272, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=412)
        self.fc4 = nn.Linear(in_features=412, out_features=2)
        
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(F.max_pool2d(self.c1(x), 3))
        x = F.relu(F.max_pool2d(self.c2(x), 3))
        x = F.relu(F.max_pool2d(self.c3(x), 3))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


# In[5]:


model = Net()


# In[6]:


model.cuda()


# In[7]:



class Settings():
    def __init__(self):
        self.data_folder = 'chest_xray/val'
        self.batch_size = 128
        self.shuffle = True
        self.lr = 0.0005
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.num_of_epochs = 10
    
    def transform_image(self):
        return transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    def loader(self):
        return torch.utils.data.DataLoader(datasets.ImageFolder(self.data_folder, 
                                                        transform=self.transform_image()),
                                                        batch_size=self.batch_size, shuffle=self.shuffle)                    
                   
    def train(self):
        ## Define forward behavior
        for epoch in range(self.num_of_epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(self.loader()):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = self.loss(output, target)
                text.insert(tk.INSERT, '\n' + 'Training loss: {:.6f}'.format(loss.item()))
                root.update()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            model.eval()
            print('Epoch: ', epoch)
            total_correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(self.loader()):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = self.loss(output, target)
                text.insert(tk.INSERT, '\n' + 'Validation loss: {:.2%}'.format(loss.item()))
                root.update()
                print('Loss: ', loss.item())
                max_arg_output = torch.argmax(output, dim=1)
                total_correct += int(torch.sum(max_arg_output == target))
                total += data.shape[0]
            text.insert(tk.INSERT, '\n' + 'Validation accuracy: {:.0%}'.format(total_correct/total))
            root.update()
            if total_correct/total > 0.8:
                torch.save(model.state_dict(), 'pt/XRP_' + str(time.strftime("%Y%m%d_%H%M%S"))+'.pt')

settings = Settings()

def start_training(): # this function will run on button press
    try:
        settings.batch_size = int(entry1.get())
        settings.lr = float(entry2.get())
        settings.num_of_epochs = int(entry3.get())
    except ValueError:
        tk.messagebox.showinfo("Error", "Please enter number.")
        return 0

    root.after(10, settings.train)


def about():
   filewin = tk.Toplevel(root)
   filewin.title('Snake')
   filewin.geometry('200x200')
   about_text='\n\n\n@andreiliphd for SPAIC\nWritten in Tkinter\n by Andrei Li\n' \
              'Saint Petersburg\n Russia \n 7 July 2019'
   text_about = tk.Label(filewin, text=about_text)


   text_about.pack()

def help():
   webbrowser.open_new('https://github.com/andreiliphd/snake')

def browse_button():
    filename = filedialog.askdirectory()
    settings.data_folder = filename
    return filename


root = tk.Tk()
frame0 = tk.Frame(root, bg='#00ffcc')
frame0.pack(side='top')

menubar = tk.Menu(frame0)

filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Open folder", command=browse_button)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

menubar.add_cascade(label="File", menu=filemenu)

helpmenu = tk.Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index", command=help)
helpmenu.add_command(label="About...", command=about)

menubar.add_cascade(label="Help", menu=helpmenu)




frame1 = tk.Frame(root, bg='#00ffcc')
frame1.pack(side='top', expand=True, fill='both')

button1 = tk.Button(frame1, text='Train', command=start_training)
button1.place(relwidth=0.3, relheight=0.3, relx=0.6, rely=0.35)

inside_frame = tk.Frame(frame1, bg='#00ffcc')
inside_frame.place(relwidth=0.3, relheight=0.8, relx=0.2, rely=0.2)

inside_frame_1 = tk.Frame(frame1, bg='#00ffcc')
inside_frame_1.place(relwidth=0.27, relheight=0.13, relx=0.2, rely=0.3)

text1 = tk.Label(inside_frame_1, bg='#00ffcc', text='Batch size')
text1.pack(side='left')

entry1 = tk.Entry(inside_frame_1, width = 10)
entry1.pack(side='right')

inside_frame_2 = tk.Frame(frame1, bg='#00ffcc')
inside_frame_2.place(relwidth=0.27, relheight=0.13, relx=0.2, rely=0.5)

text2 = tk.Label(inside_frame_2, bg='#00ffcc', text='LR')
text2.pack(side='left')

entry2 = tk.Entry(inside_frame_2,width = 10)
entry2.pack(side='right')


inside_frame_3 = tk.Frame(frame1, bg='#00ffcc')
inside_frame_3.place(relwidth=0.27, relheight=0.13, relx=0.2, rely=0.7)

text3 = tk.Label(inside_frame_3, bg='#00ffcc', text='Num of epochs')
text3.pack(side='left')

entry3 = tk.Entry(inside_frame_3, width = 10)
entry3.pack(side='right')



# entry1 = tk.Entry(inside_frame_1,)
# entry1.place(relwidth=0.3, relheight=0.8, relx=0.6, rely=0.2)
#
# text1 = tk.Label(inside_frame_1, bg = 'white', text='Train')
# text1.place(relwidth=0.3, relheight=0.8, relx=0.2, rely=0.2)
#
# inside_frame_2 = tk.Frame(frame1, bg='yellow')
# inside_frame_2.place(relwidth=0.3, relheight=0.13, relx=0.2, rely=0.6)
#
# entry2 = tk.Entry(inside_frame_2,)
# entry2.place(relwidth=0.3, relheight=0.8, relx=0.6, rely=0.3)
#
# text2 = tk.Label(inside_frame_1, bg = 'white', text='Train')
# text2.place(relwidth=0.3, relheight=0.8, relx=0.2, rely=0.2)


frame2 = tk.Frame(root, bg='#00ffcc')
frame2.pack(side='top', expand=True, fill='both')

# label_text = tk.StringVar()
scrollbar = tk.Scrollbar(frame2)
scrollbar.pack(side='right', fill='y')

text = tk.Text(frame2, bg = 'white', yscrollcommand=scrollbar.set)
text.insert(tk.INSERT, 'Training has not started yet.')
text.place(relwidth=0.9, relheight=0.9, relx=0.05, rely=0.05)

root.title('Snake')
root.geometry('640x480')
root.config(menu=menubar)


root.mainloop()

