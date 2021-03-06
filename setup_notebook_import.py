# based on https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Importing%20Notebooks.html
import ast, astor, io, os, sys, types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell


def find_notebook(fullname, path=None):
    """find a notebook, given its fully qualified name and an optional path

    This turns "foo.bar" into "foo/bar.ipynb"
    and tries turning "Foo_Bar" into "Foo Bar" if Foo_Bar
    does not exist.
    """
    name = fullname.rsplit(".", 1)[-1]
    if not path:
        path = [""]
    for d in path:
        nb_path = os.path.join(d, name + ".ipynb")
        if os.path.isfile(nb_path):
            return nb_path
        # let import Notebook_Name find "Notebook Name.ipynb"
        nb_path = nb_path.replace("_", " ")
        if os.path.isfile(nb_path):
            return nb_path


class NotebookLoader(object):
    """Module Loader for Jupyter Notebooks"""

    def __init__(self, path=None):
        self.shell = InteractiveShell.instance()
        self.path = path

    def load_module(self, fullname):
        """import a notebook as a module"""
        path = find_notebook(fullname, self.path)

        # load the notebook object
        with io.open(path, "r", encoding="utf-8") as f:
            nb = read(f, 4)

        mod = types.ModuleType(fullname)
        mod.__file__ = path
        mod.__loader__ = self
        mod.__dict__["get_ipython"] = get_ipython
        sys.modules[fullname] = mod

        # extra work to ensure that magics that would affect the user_ns
        # actually affect the notebook module's ns
        save_user_ns = self.shell.user_ns
        self.shell.user_ns = mod.__dict__

        try:
            for cell in nb.cells:
                if cell.cell_type == "code":
                    # transform the input to executable Python
                    code = self.shell.input_transformer_manager.transform_cell(
                        cell.source
                    )
                    # don't run the imported code when it is imported, because jupyter notebooks usually take fairly
                    # long to run completely. Instead save cells that can be run separately using exec.
                    ast_tree = ast.parse(code)
                    for branch in ast_tree.body:
                        # do not run top level code
                        if type(branch) in [
                            ast.Import,
                            ast.ImportFrom,
                            ast.FunctionDef,
                            ast.ClassDef,
                        ]:
                            try:
                                exec(astor.to_source(branch), mod.__dict__)
                            except (NameError, NotImplementedError):
                                # Some jupyter functions are not implemented
                                pass

        finally:
            self.shell.user_ns = save_user_ns
        return mod


class NotebookFinder(object):
    """Module finder that locates Jupyter Notebooks"""

    def __init__(self):
        self.loaders = {}

    def find_module(self, fullname, path=None):
        nb_path = find_notebook(fullname, path)
        if not nb_path:
            return

        key = path
        if path:
            # lists aren't hashable
            key = os.path.sep.join(path)

        if key not in self.loaders:
            self.loaders[key] = NotebookLoader(path)
        return self.loaders[key]

    @classmethod
    def setup(cls):
        sys.meta_path.append(cls())
