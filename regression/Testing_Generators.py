import numpy as np
from collections.abc import Iterator, Generator
import sys

BATCH_SIZE = 3000

def gen_helper(batch_size, X, Y):
    ''' can significantly improve with list comprehension '''
    num_tot = X.shape[0]
    if num_tot != Y.shape[0]:
        raise ValueError('Data input sizes mis-matched for generator')
        
    theList = []
    nbatch = np.ceil(num_tot/batch_size).astype(int)
    
    for i in range(nbatch):
        low = i*batch_size
        high = (i+1)*batch_size
        if high > num_tot:
            theList.append((low,num_tot))
            break
        else:
            theList.append((low,high))
            
    return theList


def data_gen(batch_size, X, Y, epochs):
    
    slices  = gen_helper(batch_size, X, Y)
    
    for i in range(epochs):
        
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        Xnew = X[indices,:,:]
        Ynew = Y[indices]
        
        for (c, (i,j)) in enumerate(slices, 1):
            if c == len(slices):
                print('last batch: {} -- indices: {} -- shape: {}'.format(c, (i,j), Xnew[i:j,:,:].shape[0]) );print()
            else:
                print('batch: {} -- indices: {} -- shape: {}'.format(c, (i,j), Xnew[i:j,:,:].shape[0]) )
            yield (Xnew[i:j,:,:], Ynew[i:j])

            
class DataGenerator():
    def __init__(self, batch_size, X, Y, epochs):
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.epochs = epochs
        
    def __iter__(self):
        return self
        
    def __next__(self):
        print('next called')
        slices = gen_helper(self, self.batch_size, self.X, self.Y)
        
        for i in range(self.epochs):
            print('epoch_number: {}'.format(i+1))
            
            #shuffle datas
            indices = np.arange(self.X.shape[0], dtype=int)
            np.random.shuffle(indices)
            Xnew = self.X[indices,:,:]
            Ynew = self.Y[indices]
            
            for (c, (i,j)) in enumerate(slices, 1):
                print('yield called')
                yield (Xnew[i:j,:,:], Ynew[i:j])
                
class ML_NumpyGenerator(Generator):
        def __init__(self, batch_size, X, Y, epochs):
            self.batch_size = batch_size
            self.X = X
            self.Y = Y
            self.epochs = epochs
        
        def __next__():
            pass
        
        def send(self):
            return None
        
        def throw(self, type=None, value=None, traceback=None):
            raise StopIteration
    
    
            

##################
## TESTING AREA ##
##################
X = np.zeros((50000,50,3))
Y = np.zeros(50000)
BATCH_SIZE = 3000
EPOCHS = 10

## Meta
train_num = int(.7*X.shape[0])
val_num = int(.15*X.shape[0])
print('train number: {}'.format(train_num))
steps_per_train = np.ceil(train_num/BATCH_SIZE).astype(int)
print('steps per train: {}'.format(steps_per_train))

print('calling function')
func_gen = data_gen(batch_size=BATCH_SIZE, X=X[:train_num,:,:], Y=Y[:train_num], epochs=EPOCHS)

# print('creating generator object')
# data_gen = DataGenerator(batch_size=BATCH_SIZE, X=X, Y=Y, epochs=EPOCHS)


print('looping function')
for i in range(EPOCHS):
    
    print('epoch number {}'.format(i+1))
    for j in range(steps_per_train):
        print('batch_number {}'.format(j+1))
        X_mini = next(func_gen)
        
    print()
    
    # print()
    # print('generators')
    # x = data_gen.__next__()
    # print(type(next(data_gen)))
        