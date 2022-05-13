import random

def getRandomnum(bits):
    value_set= ""
    for i in range(bits):
       value_set+= str(random.randint(0, 10))


    return value_set

if __name__ == '__main__' :
    strings = getRandomnum(10000)
    print(strings)
