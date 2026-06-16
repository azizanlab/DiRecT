import os

D3IL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
)


def d3il_path(*args) -> str:
    return os.path.abspath(os.path.join(D3IL_DIR, *args))
