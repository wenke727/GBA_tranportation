import time

def timer(func):
    def inner():
        before = time.time()
        func()
        after = time.time()
        duration = round((after - before), 1)
        print('function {} cost {} seconds'.format(func.__name__, duration))
    return inner