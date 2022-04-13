#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
uintType = np.uint8
floatType = np.float32

class HOP(object):
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N, N), dtype = floatType)

    def kroneckerSquareProduct(self, factor):
        ksProduct = np.zeros((self.N, self.N), dtype = floatType)
        for i in range(0, self.N):
            ksProduct[i] = factor[i] * factor
        return ksProduct


    def trainOnce(self, inputArray):
        mean = float(inputArray.sum()) / inputArray.shape[0]
        self.W = self.W + self.kroneckerSquareProduct(inputArray - mean) / (self.N * self.N) / mean / (1 - mean)
        index = range(0, self.N)
        self.W[index, index] = 0.


    def hopTrain(self, stableStateList):
        stableState = np.asarray(stableStateList, dtype = uintType)
        
        if len(stableState.shape) == 1 and stableState.shape[0] == self.N:
            print ('stableState count: 1')
            self.trainOnce(stableState)
        elif len(stableState.shape) == 2 and stableState.shape[1] == self.N:
            print ('stableState count: ' + str(stableState.shape[0]) )
            for i in range(0, stableState.shape[0]):
                self.trainOnce(stableState[i])
        else:
            print ('SS Dimension ERROR! Training Aborted.')
            return
        print ('Hopfield Training Complete.')

        
    def hopRun(self, inputList):
        inputArray = np.asarray(inputList, dtype = floatType)
        
        matrix = np.tile(inputArray, (self.N, 1))
        matrix = self.W * matrix
        ouputArray = matrix.sum(1)

        m = float(np.amin(ouputArray))
        M = float(np.amax(ouputArray))
        ouputArray = (ouputArray - m) / (M - m)
        ouputArray[ouputArray < 0.5] = 0.
        ouputArray[ouputArray > 0] = 1.
        return np.asarray(ouputArray, dtype = uintType)


# In[36]:


import numpy as np
import tkinter as tk
from tkinter import filedialog,dialog
import os

window = tk.Tk()
window.title('Hopfield')
window.geometry('800x700')
window.configure(background='white')

def printFormat(vector, NperGroup):
    string = ''
    for index in range(len(vector)):
        if index % NperGroup == 0:
            string += '\n'
        if str(vector[index]) == '0':
            string += ' '
        elif str(vector[index]) == '1':
            string += '*'
        else:
            string += str(vector[index])
    string += '\n'
    text1.insert('insert',string)
    print (string)
    
def counting(file_path):#計算圖形大小
    global row 
    row = 0
    global column 
    column = 0
    f = open(file_path)
    flag = 0
    for i in f.read():
        if i == ' ':
            column += 1
            flag = 0
        elif i == '\n':
            
            if flag == 1:
                break
            row += 1
            flag = 1
        else:
            column += 1
            flag = 0
    column = int(column / row)
    print(row,column)
    
def train_file():#訓練集
    file_path = filedialog.askopenfilename(title=u'選擇檔案',initialdir=(os.path.expanduser('H:/')))
    counting(file_path)
    inputs = []
    inputs.append([])
    index = 0#第幾個圖形
    counter = 0#計算行數
    f = open(file_path)
    for i in f.read():
        if i == ' ':
            inputs[index].append(0)
        elif i == '\n':
            counter += 1
            if counter % (row+1) == 0:#下一個圖形
                inputs.append([])
                index += 1
        else:
            inputs[index].append(i)
    train_data = inputs
    global hop
    hop = HOP(row * column)
    hop.hopTrain(train_data)
    text1.insert('insert','Training Complete\n')
    
def test_file():#測試集
    file_path = filedialog.askopenfilename(title=u'選擇檔案',initialdir=(os.path.expanduser('H:/')))
    text1.insert('insert','Testing Data：\n')
    inputs = []
    inputs.append([])
    counter = 0
    index = 0
    f = open(file_path)
    for i in f.read():
        if i == ' ':
            inputs[index].append(0)
        elif i == '\n':
            counter += 1
            if counter % (row+1) == 0:
                inputs.append([])
                index += 1
        else:
            inputs[index].append(i)
    test_data = inputs
    for i in range(len(test_data)):
        text1.insert('insert','Original:')
        print("Original：")
        printFormat(test_data[i], column)
        result = hop.hopRun(test_data[i])
        text1.insert('insert','Recovered:')
        print("Recovered：")
        printFormat(result, column)
    

bt1 = tk.Button(window,text='Training Data',width=15,height=2,command=train_file)
bt1.pack()
bt2 = tk.Button(window,text='Testing Data',width=15,height=2,command=test_file)
bt2.pack()
text1 = tk.Text(window,width=50,height=30,bg='white',font=('Lucida Grande',15))
text1.pack()
button = tk.Button(master=window, text="Quit",command=window.destroy)
button.pack(side=tk.BOTTOM)
window.mainloop()

