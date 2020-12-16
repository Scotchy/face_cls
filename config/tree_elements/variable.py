from .node import Node

__all__ = ["Variable"]

class Variable(Node):

    def __init__(self, name, value):
        super(Variable, self).__init__(name, value)

    def _construct(self, name, value):
        self.name = name
        self.value = value
    
    def _check_valid(self, name, value):
        return True

    def __call__(self):
        return self.value
    