from contextlib import contextmanager
from typing import Any, List


@contextmanager
def table(cols: str, caption=None, label=None, scale=False):
    print("\\begin{table}[]")

    if scale:
        print("\\resizebox{\\textwidth}{!}{")

    print(f"\\begin{{tabular}}{{ {cols} }}")
    yield

    print("\\end{tabular}")

    if scale:
        print("}")
    if caption is not None:
        print(f"\\caption{{ {caption} }}")
    if label is not None:
        print(f"\\label{{ {label} }}")
    print("\\end{table}")


def hline():
    print("\hline")


def table_row(content: List[Any], hline=True):
    s = " & ".join(content)
    s += " \\\\"
    if hline:
        s += " \hline"
    print(s)


def table_headers(content: List[Any], hline=True):
    header_content = [f"\\multicolumn{{1}}{{|c|}}{{ \\textbf{{ {content[0]} }} }}"]
    header_content += [
        f"\\multicolumn{{1}}{{c|}}{{ \\textbf{{ {text} }} }}" for text in content[1:]
    ]

    table_row(header_content, hline=hline)
