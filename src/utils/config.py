import yaml
from types import SimpleNamespace


def parse_config(path=None):
    """
    Reads a configuration file in YAML format and converts it into a namespace object.

    This function loads the configuration from the provided file path or a default
    path if none is specified. The configuration content is parsed into a Python
    dictionary and then transformed into a namespace object, allowing attribute-style
    access to the configuration data.

    :param path: The optional file path to the configuration YAML file. If not
        specified, a default path "default.yaml" is used.
    :type path: str, optional
    :return: A namespace object containing the configuration data parsed from the
        YAML file.
    :rtype: types.SimpleNamespace
    """
    if not path:
        path = "../default.yaml"

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = config_dict_to_namespace(config_dict)

    return config


def config_dict_to_namespace(d):
    """
    Recursively converts a dictionary into a SimpleNamespace object.
    If an element is a dict, it is converted.
    If it is a list, each item is checked similarly.
    Otherwise, it is returned as-is.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: config_dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [config_dict_to_namespace(item) for item in d]
    else:
        return d
