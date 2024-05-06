# -*- coding: utf-8 -*-
"""
This is a program for configuration loading.
Author: Guanbao Liang
License: BSD 2 clause
"""

import argparse
import yaml


def load_config(config_type="yaml", config_file="configs/base.yaml", parser=None):
    """
    Function loads the configuration file to set the experiment environment.

    Parameters
    ----------
    config_type : str
        The configuration file type.
    config_file : str
        The configuration file.
    parser : argparse.ArgumentParser | None
        The module simplifies command-line argument handling.

    Returns
    -------
    parser : argparse.ArgumentParser
        The module simplifies command-line argument handling and contains the arguments of configuration file.
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key, items in config.items():
        if parser.get_default(key) is not None:
            parser.set_defaults(**{key: items["default"]})

        if isinstance(items, dict):
            arg_name = "--" + str(key)
            arg_type = items.get("type", None)
            arg_type = eval(arg_type) if arg_type is not None else arg_type
            arg_default = items.get("default", None)
            arg_default = (
                arg_type(arg_default) if arg_default is not None else arg_default
            )
            arg_help = items.get("help", None)
            parser.add_argument(
                arg_name, type=arg_type, default=arg_default, help=arg_help
            )

        else:
            parser.add_argument(
                "--" + str(key), type=type(items), default=items, help=str(key)
            )

    return parser


def load_config_list(
    config_type="yaml", config_file_list=["configs/base.yaml"], parser=None
):
    """
    Function loads the configuration files to set the experiment environment.

    Parameters
    ----------
    config_type : str
        The configuration file type.
    config_file_list : List[str]
        The configuration file list.
    parser : argparse.ArgumentParser | None
        The module simplifies command-line argument handling.
    
    Returns
    -------
    parser : argparse.ArgumentParser
        The module simplifies command-line argument handling and contains the arguments of configuration files.
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    for config_file in config_file_list:
        load_config(config_type, config_file, parser)
    return parser
