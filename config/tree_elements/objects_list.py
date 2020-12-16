from .single_object import SingleObject
from .node import Node

class ObjectsList(Node):

    def __init__(self, name, config_dict):
        """Create a list of SingleObject from a yaml configuration file.

        Args:
            name (str): Name of the list of objects
            config_dict (dict): A list of dictionaries which defines the list. Its format is 
            [
                {'obj:class_name': {'param1': value, ...}}
                ...
            ]
        """
        super(ObjectsList, self).__init__(name, config_dict)

    def _check_valid(self, name, config_dict): 
        return True

    def _construct(self, name, config_dict):
        self.name = name
        self.objects = [SingleObject(name, config_dict) for config_dict in config_dict]
        
    def __call__(self, **args):
        return [obj(**args) for obj in self.objects]
