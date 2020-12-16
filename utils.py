from easydict import EasyDict as edict
import yaml

def get_conf(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_conf = yaml.load(f, Loader=yaml.Loader)
    return edict(yaml_conf)