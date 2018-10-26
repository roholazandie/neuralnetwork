import numpy as np

class Loss(object):

    @staticmethod
    def calculate(a, y):
        raise NotImplementedError("")

    @staticmethod
    def delta(grad_z, a, y):
        raise NotImplementedError("")



class EntropyLoss(Loss):

    @staticmethod
    def calculate(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(grad_z, a, y):
        return a-y


class L2Loss(Loss):

    @staticmethod
    def calculate(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(grad_z, a, y):
        return L2Loss.grad_loss(a,y) * grad_z

    @staticmethod
    def grad_loss(a, y):
        return (a-y)