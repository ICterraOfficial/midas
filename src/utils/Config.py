import re
from types import SimpleNamespace
from pprint import pformat
import yaml


__all__ = ['Config']


class NestedNamespace(SimpleNamespace):
    # Implemented from https://stackoverflow.com/a/54332748/13080899 .
    def __init__(self, dictionary=None, **kwargs):
        super(NestedNamespace, self).__init__(**kwargs)
        if dictionary:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    self.__setattr__(key, NestedNamespace(value))
                else:
                    self.__setattr__(key, value)
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = NestedNamespace(value)
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        """Return a prettified dictionary-like representation of the instance."""
        return pformat(self._to_dict(), compact=True)

    def _to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, NestedNamespace):
                result[key] = value._to_dict()
            else:
                result[key] = value
        return result


class Config(NestedNamespace):
    """
    This class is used to create a configuration object that allows access to YAML file attributes
    using dot notation.

    This class extends the `NestedNamespace` and is designed to read and parse YAML configuration files,
    enabling easier access to configuration parameters as object attributes. It is particularly useful for
    handling configuration parameters in deep learning experiments, where parameters can be structured in
    nested dictionaries.

    Attributes
    ----------
    Any YAML file attribute can be accessed as a dot notation attribute after being parsed by the
    `NestedNamespace` class.
    """

    def __init__(self, config_path):
        """Creates a config object to access yaml attributes with dot notation.

        Parameters
        ----------
        config_path: str
        """
        # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number

        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        with open(config_path) as file:
            try:
                super(Config, self).__init__(yaml.load(file, Loader=loader))
            except yaml.YAMLError as exc:
                print(exc)

    def save(self, config_path):
        """Saves the current config object to a YAML file.

        Parameters
        ----------
        config_path: str
        """
        with open(config_path, 'w') as file:
            yaml.dump(self._to_dict(), file)

    def to_markdown(self):
        """
            Convert a nested dictionary to a Markdown string with proper formatting.

            Returns
            -------
                str
                    The Markdown formatted string representing the dictionary.

        """

        d = self._to_dict()

        def dict_to_markdown(d, indent=0):
            """
            Convert a nested dictionary to a Markdown string with proper formatting.

            Parameters
            ----------
                d : dict
                    The dictionary to convert.
                indent : int
                    The current level of indentation (used for recursion).

            Returns
            -------
                str
                    The Markdown formatted string representing the dictionary.
            """

            markdown_str = ""
            indent_str = " " * (indent * 2)  # 2 spaces per indent level

            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, dict):
                        markdown_str += f"{indent_str}- **{key}:**\n"
                        markdown_str += dict_to_markdown(value, indent + 1)  # Increase indent for nested dict
                    elif isinstance(value, list):
                        markdown_str += f"{indent_str}- **{key}:**\n"
                        for item in value:
                            if isinstance(item, dict):
                                markdown_str += dict_to_markdown(item, indent + 1)  # Increase indent for dicts in list
                            else:
                                markdown_str += f"{indent_str}  - `{item}`\n"
                    else:
                        markdown_str += f"{indent_str}- **{key}:** `{value}`\n"
            else:
                markdown_str += f"{indent_str}- `{str(d)}`\n"

            return markdown_str

        return dict_to_markdown(d)
