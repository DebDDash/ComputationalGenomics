import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import os
import numpy as np
import pandas as pd
import datetime
import csv
import argparse
import shutil


# Hyperparameter Defaults

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str,
                    required=True, help="Image_Name")
parser.add_argument("-t", "--tcn", type=int, default=11, help="Num_TCN")
parser.add_argument("-r", "--runs", type=int, default=5, help="Num_Run")
parser.add_argument("-e", "--epochs", type=int, default=2500, help="Num_Epoch")
parser.add_argument("-d", "--embed", type=int, default=128,
                    help="Embedding Dimensionality")
parser.add_argument("-a", "--lr", type=float,
                    default=0.001, help="Learning_Rate")
args = parser.parse_args()

Image_Name = args.image
Num_TCN = args.tcn
Num_Run = args.runs
Num_Epoch = args.epochs
Learning_Rate = args.lr
Embedding_Dimension = args.embed
Loss_Cutoff = -0.6
InputFolderName = "./data/"


# Import image name list.
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
    Region_filename,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["Image"],  # set our own names for the columns
)


# Load dataset from the constructed Dataset.
LastStep_OutputFolderName = "./Step1_Output/"
MaxNumNodes_filename = LastStep_OutputFolderName + "MaxNumNodes.txt"
max_nodes = np.loadtxt(MaxNumNodes_filename,
                       dtype='int64', delimiter="\t").item()
final_loss = 0


class SpatialOmicsImageDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SpatialOmicsImageDataset, self).__init__(
            root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SpatialOmicsImageDataset.pt']

    def download(self):
        pass

    def process(self):
        # Read data_list into huge `Data` list.
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = SpatialOmicsImageDataset(
    LastStep_OutputFolderName, transform=T.ToDense(max_nodes))


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=Embedding_Dimension):
        super(Net, self).__init__()

        self.conv1 = DenseGraphConv(in_channels, hidden_channels)
        num_cluster1 = Num_TCN  # This is a hyperparameter.
        self.pool1 = Linear(hidden_channels, num_cluster1)

    def forward(self, x, adj, mask=None):

        x = F.relu(self.conv1(x, adj, mask))
        s = self.pool1(x)  # Here "s" is a non-softmax tensor.
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
        # Save important clustering results_1.
        ClusterAssignTensor_1 = s
        ClusterAdjTensor_1 = adj

        return F.log_softmax(x, dim=-1), mc1, o1, ClusterAssignTensor_1, ClusterAdjTensor_1


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, mc_loss, o_loss, _, _ = model(data.x, data.adj, data.mask)
        loss = mc_loss + o_loss
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all


# Extract a single graph for TCN learning.
ThisStep_OutputFolderName = "./Step2_Output_" + Image_Name + "/"
os.makedirs(ThisStep_OutputFolderName, exist_ok=True)

train_index = [region_name_list["Image"].values.tolist().index(Image_Name)]
train_dataset = dataset[train_index]
train_loader = DenseDataLoader(train_dataset, batch_size=1)
all_sample_loader = DenseDataLoader(train_dataset, batch_size=1)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
run_number = 1
fails = 0
while run_number <= Num_Run:  # Generate multiple independent runs for ensemble.

    print(f"This is Run{run_number:02d}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, 1).to(device)  # Initializing the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)

    RunFolderName = ThisStep_OutputFolderName + "Run" + str(run_number)
    if os.path.exists(RunFolderName):
        shutil.rmtree(RunFolderName)
    os.makedirs(RunFolderName)  # Creating the Run folder.

    filename_0 = RunFolderName + "/Epoch_UnsupervisedLoss.csv"
    headers_0 = ["Epoch", "UnsupervisedLoss"]
    with open(filename_0, "w", newline='') as f0:
        f0_csv = csv.writer(f0)
        f0_csv.writerow(headers_0)

    previous_loss = float("inf")  # Initialization.
    # Specify the number of epoch in each independent run.
    for epoch in range(1, Num_Epoch+1):
        train_loss = train(epoch)

        # print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")
        with open(filename_0, "a", newline='') as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow([epoch, train_loss])

        # If two consecutive losses are both zeros, the learning gets stuck.
        if train_loss == 0 and train_loss == previous_loss:
            break  # stop the training.
        else:
            previous_loss = train_loss

    print(f"Final train loss is {train_loss:.4f}")
    # This is an empirical cutoff of the final loss to avoid underfitting.
    if train_loss >= Loss_Cutoff:
        # Remove the specific folder and all files inside it for re-creating the Run folder.
        shutil.rmtree(RunFolderName)
        fails += 1
        if fails >= 3:
            print(
                "Convergence not reached. Consider changing hyperparameters before trying again.")
            break
        continue  # restart this run.
    final_loss = min(final_loss, train_loss)

    # Extract the soft TCN assignment matrix using the trained model.
    for EachData in all_sample_loader:
        EachData = EachData.to(device)
        TestModelResult = model(EachData.x, EachData.adj, EachData.mask)

        ClusterAssignMatrix1 = TestModelResult[3][0, :, :]
        # Checked, consistent with the built-in function "dense_mincut_pool".
        ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)
        ClusterAssignMatrix1 = ClusterAssignMatrix1.detach().numpy()
        filename1 = RunFolderName + "/TCN_AssignMatrix1.csv"
        np.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')

        ClusterAdjMatrix1 = TestModelResult[4][0, :, :]
        ClusterAdjMatrix1 = ClusterAdjMatrix1.detach().numpy()
        filename2 = RunFolderName + "/TCN_AdjMatrix1.csv"
        np.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')

        NodeMask = EachData.mask
        NodeMask = np.array(NodeMask)
        filename3 = RunFolderName + "/NodeMask.csv"
        # save as integers.
        np.savetxt(filename3, NodeMask.T, delimiter=',', fmt='%i')

    run_number = run_number + 1

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Collect hyperparameters and final loss
row = [Image_Name, Num_TCN, Num_Run, Num_Epoch,
       Embedding_Dimension, Learning_Rate, Loss_Cutoff, final_loss]

results_file = f"Step2_Output_{Image_Name}/Results.csv"
file_exists = os.path.isfile(results_file)
# Append the new row to the Results.csv file
with open(results_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        # Write the header if the file is being created
        header = ['Image_Name', 'Num_TCN', 'Num_Run', 'Num_Epoch',
                  'Embedding_Dimension', 'Learning_Rate', 'Loss_Cutoff', 'Final_Loss']
        writer.writerow(header)
    writer.writerow(row)
