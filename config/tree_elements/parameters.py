from .utils import is_object, is_objects_list, is_var
from .node import Node
from .variable import Variable
import config.tree_elements.single_object as single_object
import config.tree_elements.objects_list as objects_list

__all__ = ["Parameters"]

class Parameters(Node):

    def __init__(self, class_name, param_dict):
        """Create parameters of an object from a dict 'param_dict' of format 
        { 
            object_param_name: {class_name: obj_param_dict},
            variable_param_name: value,
            objects_list_param_name: [class_name: obj_param_dict, ...]
        }

        Args:
            param_dict (dict): Dictionary of the parameters
        """
        super(Parameters, self).__init__(class_name, param_dict)
        
    def _construct(self, class_name, param_dict):
        self.class_name = class_name
        self._params = {}
        for k, param_dict in param_dict.items():
            if is_var(param_dict):
                self._params[k] = Variable(k, param_dict)
                self.__dict__[k] = Variable(k, param_dict)
            elif is_object(param_dict):
                self._params[k] = single_object.SingleObject(k, param_dict)
                self.__dict__[k] = single_object.SingleObject(k, param_dict)
            elif is_objects_list(param_dict):
                self._params[k] = objects_list.ObjectsList(class_name, param_dict)
                self.__dict__[k] = objects_list.ObjectsList(class_name, param_dict)
            else:
                raise ValueError("")

    def _check_valid(self, class_name, param_dict):
        for k, v in param_dict.items():
            if not is_var(v) and not is_object(v) and not is_objects_list(v):
                raise ValueError(f"Found bad format for parameter {k} of object {class_name} {param_dict}.")
        return True

    def unwarp(self):
        return {param_name: param_value() for param_name, param_value in self._params.items()}