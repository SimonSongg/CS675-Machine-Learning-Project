import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import argparse
from test_CSV import DealDataset

parser = argparse.ArgumentParser(description = "Alzheimer's Disease Classification: Train")

parser.add_argument('--dataset_dir', type=str, default='./data', help='Directory for storing data_set')
parser.add_argument('--model_path', type=str, default='./models/current.pth', help='Path for storing model')

args = parser.parse_args()

path = args.dataset_dir
model_path = args.model_path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_conv = torchvision.models.resnet18(pretrained=True)

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model_conv.load_state_dict(torch.load(model_path))
model_conv = model_conv.to(device)
model_conv.eval()
test_data = DealDataset(path, 'train')
test_loader = DataLoader(dataset=test_data, batch_size=1)
total_pos = 0
total_neg = 0
correct_pos = 0
correct_neg = 0
for input, label in test_loader:
    input = input.to(device)
    label = label.to(device)
    output = model_conv(input)
    _, pred = torch.max(output, 1)
    if label == 1:
        total_pos = total_pos + 1
        if pred == label:
            correct_pos = correct_pos + 1
    else:
        total_neg = total_neg + 1
        if pred == label:
            correct_neg = correct_neg + 1

print('TP:', correct_pos)
print('TN:', correct_neg)
print('Total number of positive samples:', total_pos)
print('Total number of negative samples::', total_neg)
print('Acc:', (correct_neg + correct_pos) / test_data.__len__())
print('Recall:', correct_pos / total_pos)
print('Total number of samples::', test_data.__len__())
