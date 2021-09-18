import argparse
import configparser
import datetime
import logging
import os

from utils.tool import Args

DEFAULT_CONFIGURE_DIR = "configure"
DEFAULT_DATASET_DIR = "data"
DEFAULT_MODEL_DIR = "models"
DEFAULT_EXP_DIR = "saved"
DEFAULT_CONSOLE_ARGS_LABEL_FILE = "__console__.cfg"


class String(object):
    @staticmethod
    def to_basic(string):
        """
        Convert the String to what it really means.
        For example, "true" --> True as a bool value
        :param string:
        :return:
        """
        try:
            return int(string)
        except ValueError:
            try:
                return float(string)
            except ValueError:
                pass
        if string in ["True", "true"]:
            return True
        elif string in ["False", "false"]:
            return False
        else:
            return string


class Configure(object):
    @staticmethod
    def get_file_cfg(file):
        """
        get configurations in file.
        :param file:
        :return: configure args
        """
        cfgargs = Args()
        parser = configparser.ConfigParser()
        parser.read(file)
        for section in parser.sections():
            setattr(cfgargs, section, Args())
            for item in parser.items(section):
                setattr(getattr(cfgargs, section), item[0], String.to_basic(item[1]))
        return cfgargs

    @staticmethod
    def get_console_cfg(default_file):
        """
        get configurations from console.
        :param default_file:
        :return:
        """
        conargs = Args()
        parser = argparse.ArgumentParser()
        types = {"bool": bool, "int": int, "float": float}
        args_label = Configure.get_file_cfg(default_file)
        for arg_name, arg in args_label:
            argw = {}
            if arg.help:
                argw["help"] = arg.help
            if arg.type == "implicit_bool" or arg.type == "imp_bool":
                argw["action"] = "store_true"
            if arg.type == "string" or arg.type == "str" or arg.type is None:
                if arg.default:
                    if arg.default == "None" or "none":
                        argw["default"] = None
                    else:
                        argw["default"] = arg.default
            if arg.type in types:
                argw["type"] = types[arg.type]
                if arg.default:
                    if arg.default == "None" or "none":
                        argw["default"] = None
                    else:
                        argw["default"] = types[arg.type](arg.default)
            parser.add_argument("--" + arg_name, **argw)
        tmpargs = parser.parse_args()
        for arg_name, arg in args_label:
            setattr(conargs, arg_name, getattr(tmpargs, arg_name))
        return conargs

    @staticmethod
    def Get():
        conargs = Configure.get_console_cfg(os.path.join(DEFAULT_CONFIGURE_DIR, DEFAULT_CONSOLE_ARGS_LABEL_FILE))
        logging.info("Loading configure from " + conargs.cfg)
        args = Configure.get_file_cfg(os.path.join(DEFAULT_CONFIGURE_DIR, conargs.cfg))
        if conargs.debug:
            logging.debug("Debug flag found")
            for arg_name, arg in args.debug:
                cur = args
                arg_divs = arg_name.split(".")
                for arg_div in arg_divs[: -1]:
                    cur = getattr(cur, arg_div)
                setattr(cur, arg_divs[-1], arg)
                delattr(args.debug, arg_name)
            args.debug = True
        if not args.model.nick:
            args.model.nick = args.model.name + "," + str(datetime.datetime.now()).replace(":", ".").replace(" ", ",")[
                                                      0:19]
        if args.dir is not Args:
            args.dir = Args()
        args.dir.model = DEFAULT_MODEL_DIR
        args.dir.exp = DEFAULT_EXP_DIR
        args.dir.dataset = DEFAULT_DATASET_DIR
        args.dir.configure = DEFAULT_CONFIGURE_DIR
        args.dir.output = os.path.join(args.dir.exp, args.model.nick)
        for arg_name, arg in conargs:
            if arg is None:
                continue
            if arg_name != "cfg":
                names = arg_name.split(".")
                cur = args
                for name in names[: -1]:
                    if getattr(cur, name) is None:
                        setattr(cur, name, Args())
                    cur = getattr(cur, name)
                setattr(cur, names[-1], arg)
        return args
