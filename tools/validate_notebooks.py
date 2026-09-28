"""Structural checks on the notebooks, run in CI on every push.

A notebook that parses is not a notebook that will run. These are the four
failures this project has actually shipped, each now a test:

1. A cell that does not parse. Cheap to catch, expensive to discover eight
   hours into a training run.
2. A `config.X` reference that no longer exists. Constants get renamed in
   `src/config.py` and the notebook keeps the old spelling until it runs.
3. A missing contract header. The README promises every notebook opens with an
   H1 and a Reads/Writes/Runtime/Requires table; without a check that promise
   decays silently.
4. A name read before anything binds it. Notebooks execute top to bottom, so a
   name loaded in cell N must be bound by cell <= N. This is the check that
   catches a variable renamed in one cell and not another -- the class of bug
   that only shows up on a clean Run All, which is exactly the run nobody does
   before committing.

Exit code is non-zero if anything fails, so CI goes red.

    python tools/validate_notebooks.py
"""

from __future__ import annotations

import ast
import builtins
import glob
import importlib.util
import json
import os
import re
import sys

# Injected by the Databricks runtime rather than imported, so every notebook
# reads them as free variables and a name checker must be told about them.
DATABRICKS_GLOBALS = {
    "spark", "dbutils", "display", "displayHTML", "sc", "sqlContext", "getArgument",
}
KNOWN = set(dir(builtins)) | DATABRICKS_GLOBALS

CONTRACT_MARKER = "<!-- contract -->"
MAX_LINE = 127


class CellScope(ast.NodeVisitor):
    """Names bound and names read by one cell, with function bodies deferred.

    A function body may legitimately reference a name defined in a later cell --
    `run_variant` calls helpers declared below it -- so loads inside a def are
    only reported when the name is neither an argument nor bound anywhere in the
    same function.
    """

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.loads: list[tuple[str, int]] = []

    def _bind_target(self, node: ast.AST) -> None:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                self.bound.add(n.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node) -> None:
        self.bound.add(node.name)
        args = node.args
        local = {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
        for extra in (args.vararg, args.kwarg):
            if extra:
                local.add(extra.arg)
        inner = CellScope()
        for stmt in node.body:
            inner.visit(stmt)
        for name, line in inner.loads:
            if name not in local and name not in inner.bound:
                self.loads.append((name, line))
        for decorator in node.decorator_list:
            self.visit(decorator)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        local = {a.arg for a in node.args.args}
        inner = CellScope()
        inner.visit(node.body)
        for name, line in inner.loads:
            if name not in local and name not in inner.bound:
                self.loads.append((name, line))

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads.append((node.id, node.lineno))
        else:
            self.bound.add(node.id)


def code_cells(nb: dict):
    """Code cells that Python should parse, skipping `%pip` / `!` magics."""
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith(("%", "!")):
            continue
        yield index, source


def config_constants() -> set[str]:
    spec = importlib.util.spec_from_file_location("cfg", "src/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name for name in dir(module) if name.isupper()}


def check_notebook(path: str) -> list[str]:
    problems: list[str] = []
    name = os.path.basename(path)

    with open(path, encoding="utf-8") as handle:
        nb = json.load(handle)

    if CONTRACT_MARKER not in "".join(nb["cells"][0]["source"]):
        problems.append(f"{name}: first cell has no {CONTRACT_MARKER} header")

    defined = set(KNOWN)
    for index, source in code_cells(nb):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            problems.append(f"{name} cell {index}: SyntaxError: {exc.msg} (line {exc.lineno})")
            continue

        for line_no, line in enumerate(source.splitlines(), 1):
            if len(line) > MAX_LINE:
                problems.append(
                    f"{name} cell {index} line {line_no}: {len(line)} chars > {MAX_LINE}"
                )

        scope = CellScope()
        # Statement-level binding forms whose targets the visitor would otherwise
        # only see as Load contexts.
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                scope._bind_target(node.target)
            elif isinstance(node, ast.withitem) and node.optional_vars:
                scope._bind_target(node.optional_vars)
        for stmt in tree.body:
            scope.visit(stmt)

        unknown = sorted({n for n, _ in scope.loads if n not in defined and n not in scope.bound})
        if unknown:
            problems.append(f"{name} cell {index}: name(s) used before assignment: {unknown}")
        defined |= scope.bound

    return problems


def main() -> int:
    notebooks = sorted(glob.glob("notebooks/*.ipynb"))
    if not notebooks:
        print("No notebooks found — wrong working directory?")
        return 1

    available = config_constants()
    problems: list[str] = []

    for path in notebooks:
        found = check_notebook(path)
        problems += found
        print(f"  {os.path.basename(path):<24} {'OK' if not found else f'{len(found)} problem(s)'}")

        with open(path, encoding="utf-8") as handle:
            text = "\n".join("".join(c["source"]) for c in json.load(handle)["cells"])
        missing = sorted(set(re.findall(r"config\.([A-Z][A-Z0-9_]*)", text)) - available)
        if missing:
            problems.append(f"{os.path.basename(path)}: unknown config constant(s): {missing}")

    for source in sorted(glob.glob("src/*.py")):
        with open(source, encoding="utf-8") as handle:
            missing = sorted(set(re.findall(r"config\.([A-Z][A-Z0-9_]*)", handle.read())) - available)
        if missing:
            problems.append(f"{source}: unknown config constant(s): {missing}")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"All {len(notebooks)} notebooks valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
