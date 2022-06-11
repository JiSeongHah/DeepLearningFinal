import torch
from torch.utils.data import DataLoader,Dataset,TensorDataset
import numpy as np
import pickle

# class MyFinalHomworkDataset(Dataset):
#
#     def __init__(self,
#                  baseDir,
#                  task):
#         super(MyFinalHomworkDataset,self).__init__()
#
#         self.baseDir = baseDir
#         self.task = task
#
#         if self.task == 'trn':
#             trainInputName = 'train_input.txt'
#             trainTargetName= 'train_target.txt'
#             self.inputArr = np.loadtxt(self.baseDir + trainInputName, delimiter=',')
#             self.targetArr = np.loadtxt(self.baseDir + trainTargetName)
#
#         else:
#             testInputName = 'test_input.txt'
#             self.inputArr = np.loadtxt(self.baseDir + testInputName, delimiter=',')
#
#
#     def __len__(self):
#         return len(self.inputArr)
#
#     def __getitem__(self, idx):
#
#         if self.task == 'trn':
#             inputVector = self.inputArr[idx]
#             label = self.targetArr[idx]
#
#             return inputVector,label
#
x = [1]
print(np.mean(x))
# dir = '/home/a286winteriscoming/Downloads/FinalHomwork/'
#
# lst = [str(i)+'번쨰 라인' for i in range(100)]
#
# with open(dir+'test.txt','w') as f:
#     for i in lst:
#         f.write(i+'\n')

# # Inp = torch.from_numpy(np.loadtxt(dir+'train_input.txt',delimiter=','))
# # lab = torch.from_numpy(np.loadtxt(dir+'train_target.txt'))
# Inp = torch.from_numpy(np.loadtxt(dir+'test_input.txt',delimiter=','))
#
# dt = TensorDataset(Inp)
# Dl = DataLoader(dt,batch_size=3)
#
# x  = 0
# y = 0
# for Input, in Dl:
#     print(Input.size())















