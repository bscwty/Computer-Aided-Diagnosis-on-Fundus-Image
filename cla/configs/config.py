import yaml
from easydict import EasyDict

def getConfig(config_file):
    
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    return config