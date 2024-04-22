"""
Command line argument decorator.
"""

import argparse
import functools


def command(usage: str = None, description: str = None):
    """
    Decorator that define an argument parser for the function.

    Example:
    ```Python
    @cli.command(usage="main.py --arg1 <arg1> --arg2 <arg2>", description="Do something")
    @cli.argument("arg1", help="First argument", type=int, required=True)
    @cli.argument("arg2", help="Second argument", type=str, required=True)
    def main(*_args, arg1, arg2, **_kwargs):
        pass
    ```

    :param usage: The usage string for the argument parser
    :param description: The description string for the argument parser
    """

    def decorator(func):
        if not hasattr(func, "arg_decorators"):
            func.arg_decorators = []

        func.arg_decorators.append(("command", (usage, description)))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            parser = argparse.ArgumentParser(usage=usage, description=description)
            
            arg_decorators = getattr(func, "arg_decorators", [])

            for decorator_type, decorator_args in arg_decorators:
                if decorator_type == "argument":
                    arg_name, arg_kwargs = decorator_args
                    parser.add_argument("--" + arg_name, **arg_kwargs)

            parsed_args = parser.parse_args()

            return func(*args, **{**vars(parsed_args), **kwargs})

        return wrapper

    return decorator


def argument(name, **kwargs):
    """
    Decorator for adding arguments to the argument parser.

    Example:
    ```Python
    @cli.argument("arg1", help="an argument", type=int, required=True)
    def func(*_args, arg1, **_kwargs):
        pass
    ```

    :param name: The name of the argument
    """

    def decorator(func):
        if not hasattr(func, "arg_decorators"):
            func.arg_decorators = []

        func.arg_decorators.append(("argument", (name, kwargs)))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
