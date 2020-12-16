# from .template import Template
from .utils import is_object, is_objects_list, is_var
from . import SingleObject, ObjectsList, Variable
import yaml
from .node import Node
from .parameters import Parameters
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

__all__ = ["Config"]

class Config(Node):

    def __init__(self, config_dict, name=None):
        if name == "root":
            raise ValueError("Forbidden name 'root' in yaml file.")
        if name is None:
            name = "root"
        super(Config, self).__init__(name, config_dict)
        self.config_dict = config_dict

    def _check_valid(self, name, config_dict):
        return True

    @property
    def is_root(self):
        return self.name == "root"

    def _construct(self, name, sub_config):
        for name, sub_config in self.config_dict.items():
            self.set_node(name, sub_config)

    def set_node(self, name, sub_config):
        if is_var(sub_config):
            setattr(self, name, Variable(name, sub_config))

        elif is_object(sub_config):
            
            setattr(
                self, 
                name, 
                SingleObject(
                    name, 
                    sub_config
                )
            )
        elif is_objects_list(sub_config):
            sub_configs = []
            for sc in sub_config:
                sub_name, sub_sc = list(sc.items())[0]
                sub_configs.append(Config(sub_sc, sub_name))
                
            setattr(
                self, 
                name, 
                ObjectsList(
                    sub_name, 
                    sub_configs
                )
            )
        elif isinstance(sub_config, dict):

            setattr(self, name, Config(sub_config, name)) # Create an attribute containing the config stored in 'key'
        else: 

            raise ValueError(f"Yaml file format not supported ({name} : {type(sub_config)})")

    def __str__(self):
        raise NotImplementedError()

    @staticmethod
    def from_yaml(config_file : str, template=None):
        """Load a configuration file and return an ObjectLoader which can instantiate the wanted objects.

        Args:
            config_file (str): The path of the yaml config file
            template (Template): A template containing information of how to load objects defined in the configuration file
        """
        with open(config_file, "r") as stream:
            yaml_dict = yaml.load(stream, Loader=Loader)
        return Config(yaml_dict)
