
class Normalize():

    def __init__(self):
        pass

    def __call__(self, x, y):
        x_norm = x / 255
        return x_norm, y