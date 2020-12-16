
class Node():

    def __init__(self, name, config_dict):
        self.name = name 
        self.config_dict = config_dict
        self._check_valid(name, config_dict)
        self._construct(name, config_dict)

    def _construct(self, name, config_dict):
        raise NotImplementedError("This function has to be implemented")

    def _check_valid(self, name, config_dict):
        raise NotImplementedError("This function has to be implemented")