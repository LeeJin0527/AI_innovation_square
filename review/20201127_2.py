def test_funct(input1, input2):
    result = input1 + input2
    return result

def get_mean(input_arr):
    sum_ = 0
    for cnt, val in enumerate(input_arr):
        sum_ += val
    return sum_ / (cnt+1)

def get_accuracy(labels, predictions):
    predictions = (predictions >= 0.5).astype(np.int)
    accuracy = np.sum((labels == predictions).astype(np.int))/labels.shape[0]*100
    accuracy = np.around(accuracy, 2)
    return accuracy

def get_bce(labels, predictions):
    bce_loss = -1*np.mean(labels*np.log(predictions) + \
                      (1-labels)*np.log(1-predictions))
    return bce_loss

labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([0.1, 0.9, 0.2, 0.3, 0.1, 0.6, 0.7])

accruacy = get_accuracy(labels, predictions)
bce_loss = get_bce(labels, predictions)

#%% What's return
def test_function(a, b):
    result = a + b
    return result

print(test_function(10, 20))
print(30)

def addition(a, b):
    return a + b

def subtraction(a, b):
    return a - b

c = addition(10, 20) + subtraction(10, 20)

c = addition(10, subtraction(10, 20))
# c = addition(10, -10)
# c = 0

#%% 
import numpy as np

# input x, output x
def say_hello():
    print('Hello World!')
    
# input x, output o
def get_random_number():
    random_number = np.random.normal(0, 1, size=(1, ))
    return random_number

# input o, output x
def say_hello2(name):
    print('Hello ', name)
    
# input o, output o
def get_mean(score_list):
    sum_ = 0
    for cnt, score in enumerate(score_list):
        sum_ += score
    return sum_ / (cnt + 1)

# say_hello()
# print(get_random_number())
# say_hello2('Shin')

#%% Namespace
# global namespace
a = 10
# print(locals(), '\n')

def test_function():
    # local namespace
    a = 20
    print(locals())
    # print(a)  
    
def test_function2():
    # local namespace
    a = 30
    print(locals())
    # print(a)
    
test_function()
test_function2()

#%%
def test_function(input1, input2):
    result = input1 + input2
    return result

addition = test_function(input1, input2)




















