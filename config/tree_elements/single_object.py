import importlib
from .parameters import Parameters
from .node import Node

__all__ = ["SingleObject"]

class SingleObject(Node):
    """ An object 
    """

    def __init__(self, name, config_dict):
        """Allow the instantiation of an object defined in a yaml configuration file.

        Args:
            name (str): Name of the object
            config_dict (dict): A dictionary defining the object (class name and parameters). Its format is
            {
                'obj:class_name': {'param1': value, ...},
                'module': 'a.b.c'
            }
        """
        super(SingleObject, self).__init__(name, config_dict)

    def _check_valid(self, name, config_dict):
        return True

    def _construct(self, name, config_dict):
        self.name = name
        self.module = config_dict.get("module", None)
        self.class_name, self.params = list(config_dict.items())[0]
        self.class_name = self.class_name.replace("obj:", "")
        self.params = Parameters(self.class_name, self.params)

    def __call__(self, **args):
        module = importlib.import_module(self.module)
        class_object = getattr(module, self.class_name)
        params = self.params.unwarp()
        return class_object(**params, **args)