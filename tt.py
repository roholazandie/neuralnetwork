

class sample():
    def __init__(self, x):
        if x=="x":
            self.func = lambda x: self.sine(x)



    def sine(self, x):
        if self.func == self.sine:
            return x**2

    def compute(self, y):
        return self.func(y)

if __name__ == "__main__":
    ss = sample("x")
    print(ss.compute(7))

    for i in range(2, 5):
        print(i)