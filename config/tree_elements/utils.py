
def is_objects_list(config_dict):
    if not isinstance(config_dict, list) or len(config_dict) == 0:
        return False
    for obj in config_dict:
        if not isinstance(obj, dict):
            return False
        name = list(obj.items())[0][0]
        if not isinstance(name, str) or name[0:4] != "obj:":
            return False
    return True

def is_object(config_dict):
    if not isinstance(config_dict, dict):
        return False
    config_dict_cp = config_dict.copy()
    config_dict_cp.pop("module", None)

    return len(config_dict_cp) == 1 and list(config_dict_cp.items())[0][0][0:4] == "obj:"
    
def is_var(config_dict):
    if is_objects_list(config_dict):
        return False
    return isinstance(config_dict, int) or isinstance(config_dict, float) or isinstance(config_dict, str) or isinstance(config_dict, list)