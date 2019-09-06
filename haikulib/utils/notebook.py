import importlib
import inspect

import IPython


def display_source(module: str, function: str) -> IPython.display.Code:
    """Display the source of the provided function in a notebook.

    :param module: The module containing function. Must be importable.
    :param function: The function whose source we wish to display.
    """
    __module = importlib.import_module(module)
    __methods = dict(inspect.getmembers(__module, inspect.isfunction))

    def _jupyterlab_repr_html_(self):
        from pygments import highlight
        from pygments.formatters import HtmlFormatter

        fmt = HtmlFormatter()
        style = "<style>{}\n{}</style>".format(
            fmt.get_style_defs(".output_html"), fmt.get_style_defs(".jp-RenderedHTML")
        )
        return style + highlight(self.data, self._get_lexer(), fmt)

    # Replace the _repr_html_() method with our own that also adds the `jp-RenderedHTML` class
    # to fix https://github.com/jupyterlab/jupyterlab/issues/6376.
    IPython.display.Code._repr_html_ = _jupyterlab_repr_html_
    # Mmmm. Slimy.
    return IPython.display.Code(data=inspect.getsource(__methods[function]), language="python3")
