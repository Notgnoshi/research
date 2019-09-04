import importlib
import inspect

import IPython
import pygments


def display_source(module: str, function: str) -> IPython.core.display.HTML:
    """Display the source of the provided function in a notebook.

    :param module: The module containing function. Must be importable.
    :param function: The function whose source we wish to display.
    """
    __module = importlib.import_module(module)
    __methods = dict(inspect.getmembers(__module, inspect.isfunction))

    return IPython.core.display.HTML(
        pygments.highlight(
            inspect.getsource(__methods[function]),
            pygments.lexers.PythonLexer(),
            pygments.formatters.HtmlFormatter(full=True),
        )
    )
