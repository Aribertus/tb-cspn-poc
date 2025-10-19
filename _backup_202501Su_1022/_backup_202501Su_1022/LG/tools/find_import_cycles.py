import ast
import os
from collections import defaultdict, deque


def is_package_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "__init__.py"))


def discover_modules(root: str):
    """
    Discover python modules in the repo and map file paths to module names.
    Returns:
      - file_to_module: path -> module.name
      - module_to_file: module.name -> path
      - packages: set of package module names
    """
    file_to_module = {}
    module_to_file = {}
    packages = set()

    # Find package roots by presence of __init__.py
    # We'll build module names relative to repo root.
    SKIP_DIRS = {"__pycache__", "venv", ".venv", "env", ".env", "site-packages"}
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip cache and virtualenv dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        if "__init__.py" in filenames:
            # Mark package path parts
            rel = os.path.relpath(dirpath, root)
            if rel == ".":
                # Root package (not typical) — ignore
                pass
            else:
                pkg = rel.replace(os.sep, ".")
                packages.add(pkg)

        for fn in filenames:
            if fn.endswith(".py"):
                rel_file = os.path.relpath(os.path.join(dirpath, fn), root)
                if os.path.basename(rel_file).startswith("."):
                    continue
                parts = rel_file.split(os.sep)
                # Determine module name: if inside package dirs, join with dots
                # If it's an __init__.py, the module is the package itself
                if fn == "__init__.py":
                    mod = os.path.dirname(rel_file).replace(os.sep, ".")
                    if mod == "":
                        # top-level __init__.py (unlikely); use package name
                        mod = os.path.splitext(fn)[0]
                else:
                    if len(parts) > 1 and is_package_dir(os.path.join(root, *parts[:-1])):
                        mod = ".".join(parts[:-1] + [os.path.splitext(parts[-1])[0]])
                    else:
                        mod = os.path.splitext(rel_file)[0].replace(os.sep, ".")

                file_path = os.path.join(root, rel_file)
                file_to_module[file_path] = mod
                module_to_file[mod] = file_path

    return file_to_module, module_to_file, packages


def resolve_relative(module: str, level: int, name: str | None, *, is_package: bool) -> str | None:
    """
    Resolve a relative import like from .sub import x inside `module`.
    Returns absolute module name or None if cannot resolve.
    """
    if level == 0:
        return name
    parts = module.split(".")
    # Determine the current package context
    package_parts = parts if is_package else parts[:-1]
    # level=1 means current package; level=2 means go up one, etc.
    up = max(level - 1, 0)
    if up > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - up]
    if name:
        return ".".join(base + name.split(".")) if base else name
    return ".".join(base)


def parse_imports(file_path: str, this_module: str) -> set[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return set()

    deps: set[str] = set()
    is_pkg = os.path.basename(file_path) == "__init__.py"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    deps.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            target = resolve_relative(this_module, getattr(node, "level", 0), mod or None, is_package=is_pkg)
            if target:
                deps.add(target)
            # Also consider explicit submodules in from x import y where y is a module
            if mod:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    deps.add(f"{target}.{alias.name}" if target else alias.name)
    return deps


def build_graph(root: str):
    file_to_module, module_to_file, _ = discover_modules(root)
    modules = set(module_to_file)
    graph = defaultdict(set)

    for file_path, mod in file_to_module.items():
        deps = parse_imports(file_path, mod)
        # keep only deps that exist in our project, and normalize to highest existing
        resolved = set()
        for dep in deps:
            # Reduce dep to nearest existing module in our map
            parts = dep.split(".")
            while parts:
                cand = ".".join(parts)
                if cand in module_to_file:
                    resolved.add(cand)
                    break
                parts.pop()
        for r in resolved:
            if r != mod:
                graph[mod].add(r)
    return graph, modules


def find_cycles(graph: dict[str, set[str]]):
    # Johnson's algorithm for simple cycles or use DFS to find back edges.
    # For practical use here, find strongly connected components > 1.
    index = {}
    lowlink = {}
    index_counter = [0]
    stack = []
    on_stack = set()
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, ()): 
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in graph.keys():
        if v not in index:
            strongconnect(v)

    # Filter SCCs that have edges forming cycles (size>1) or self-loop
    cycles = []
    for scc in sccs:
        if len(scc) > 1:
            cycles.append(sorted(scc))
        else:
            v = scc[0]
            if v in graph.get(v, set()):
                cycles.append([v])
    return cycles


def main():
    root = os.getcwd()
    graph, modules = build_graph(root)
    cycles = find_cycles(graph)
    if not cycles:
        print("No circular imports detected.")
        return
    print("Detected circular import components (SCCs):")
    for comp in cycles:
        print(" - " + " -> ".join(comp) + " -> " + comp[0])

    print("\nSuggestions:")
    for comp in cycles:
        print(f" - Break cycle among: {', '.join(comp)}")
        print("   Consider: move shared types/constants to a neutral module;\n"
              "   replace top-level imports with local/lazy imports;\n"
              "   invert dependency using interfaces or callbacks;\n"
              "   or merge tightly coupled modules.")


if __name__ == "__main__":
    main()
