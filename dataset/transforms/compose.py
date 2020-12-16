
class Compose():

    def apply_transforms(self, x, y):
        for transform in self.transforms:
            x, y = transform(x, y)
        return x, y
    
    def __init__(self, transforms):
        self.transforms = transforms
        if self.transforms is None:
            self.apply_transforms = lambda x, y: (x, y)

    
    def __call__(self, x, y):
        x, y = self.apply_transforms(x, y)
        return x, y