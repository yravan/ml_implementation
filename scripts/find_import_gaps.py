#!/usr/bin/env python3
"""
Comprehensive Import Gap Finder

This script systematically checks for missing imports, undefined classes,
and broken references across the entire codebase.
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import Set, Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_all_python_files(directory: Path) -> List[Path]:
    """Get all Python files in directory recursively."""
    return list(directory.rglob("*.py"))

def extract_imports_from_file(filepath: Path) -> Tuple[Set[str], Set[str]]:
    """Extract all imports from a Python file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
    except:
        return set(), set()

    from_imports = set()  # from X import Y
    direct_imports = set()  # import X

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    from_imports.add((node.module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                direct_imports.add(alias.name)

    return from_imports, direct_imports

def extract_defined_names(filepath: Path) -> Set[str]:
    """Extract all class and function definitions from a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
    except:
        return set()

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)

    return names

def extract_all_from_init(init_path: Path) -> Set[str]:
    """Extract __all__ list from __init__.py."""
    try:
        with open(init_path, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
    except:
        return set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, ast.List):
                        return {
                            elt.s if isinstance(elt, ast.Str) else elt.value
                            for elt in node.value.elts
                            if isinstance(elt, (ast.Str, ast.Constant))
                        }
    return set()

def try_import_module(module_path: str) -> Tuple[bool, str]:
    """Try to import a module and return success status and error."""
    try:
        # Convert file path to module path
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            return False, f"Module not found: {module_path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, ""
    except Exception as e:
        return False, str(e)

def check_test_imports(test_dir: Path) -> Dict[str, List[str]]:
    """Check all test files for import errors."""
    issues = {}

    for test_file in test_dir.glob("test_*.py"):
        file_issues = []
        from_imports, _ = extract_imports_from_file(test_file)

        for module, name in from_imports:
            if module.startswith('python.'):
                # Check if the name exists in the module
                module_path = PROJECT_ROOT / module.replace('.', '/')

                # Check __init__.py
                init_path = module_path / '__init__.py'
                if init_path.exists():
                    defined = extract_defined_names(init_path)
                    exported = extract_all_from_init(init_path)

                    if name not in defined and name not in exported:
                        # Check submodules
                        found = False
                        for subfile in module_path.glob("*.py"):
                            if subfile.name != '__init__.py':
                                sub_defined = extract_defined_names(subfile)
                                if name in sub_defined:
                                    found = True
                                    break

                        if not found:
                            file_issues.append(f"Missing: {module}.{name}")
                elif module_path.with_suffix('.py').exists():
                    # It's a direct .py file
                    defined = extract_defined_names(module_path.with_suffix('.py'))
                    if name not in defined:
                        file_issues.append(f"Missing: {module}.{name}")

        if file_issues:
            issues[str(test_file.name)] = file_issues

    return issues

def check_init_exports(python_dir: Path) -> Dict[str, List[str]]:
    """Check that all exports in __init__.py actually exist."""
    issues = {}

    for init_file in python_dir.rglob("__init__.py"):
        if init_file.parent == python_dir:
            continue  # Skip root __init__.py

        module_dir = init_file.parent
        exported = extract_all_from_init(init_file)
        defined_in_init = extract_defined_names(init_file)

        # Get all definitions from submodules
        all_defined = set(defined_in_init)
        for subfile in module_dir.glob("*.py"):
            if subfile.name != '__init__.py':
                all_defined.update(extract_defined_names(subfile))

        # Check subdirectories
        for subdir in module_dir.iterdir():
            if subdir.is_dir() and (subdir / '__init__.py').exists():
                sub_init = subdir / '__init__.py'
                all_defined.update(extract_defined_names(sub_init))
                all_defined.update(extract_all_from_init(sub_init))

        missing = exported - all_defined
        if missing:
            rel_path = init_file.relative_to(PROJECT_ROOT)
            issues[str(rel_path)] = list(missing)

    return issues

def check_internal_imports(python_dir: Path) -> Dict[str, List[str]]:
    """Check for broken internal imports within modules."""
    issues = {}

    for py_file in python_dir.rglob("*.py"):
        from_imports, _ = extract_imports_from_file(py_file)
        file_issues = []

        for module, name in from_imports:
            # Skip external modules
            if not module.startswith(('python.', '.')):
                continue

            # Handle relative imports
            if module.startswith('.'):
                # Convert to absolute
                parts = str(py_file.relative_to(PROJECT_ROOT)).replace('/', '.').replace('.py', '').split('.')
                if module == '.':
                    abs_module = '.'.join(parts[:-1])
                else:
                    levels = len(module) - len(module.lstrip('.'))
                    abs_module = '.'.join(parts[:-levels]) + module.lstrip('.')
                module = abs_module

            module_path = PROJECT_ROOT / module.replace('.', '/')

            found = False

            # Check as directory with __init__.py
            init_path = module_path / '__init__.py'
            if init_path.exists():
                defined = extract_defined_names(init_path)
                exported = extract_all_from_init(init_path)
                if name in defined or name in exported:
                    found = True
                else:
                    # Check submodules
                    for subfile in module_path.glob("*.py"):
                        if name in extract_defined_names(subfile):
                            found = True
                            break

            # Check as .py file
            py_path = module_path.with_suffix('.py')
            if py_path.exists():
                if name in extract_defined_names(py_path):
                    found = True

            if not found and module.startswith('python.'):
                file_issues.append(f"Cannot find: {module}.{name}")

        if file_issues:
            rel_path = py_file.relative_to(PROJECT_ROOT)
            issues[str(rel_path)] = file_issues

    return issues

def runtime_import_check(python_dir: Path) -> Dict[str, str]:
    """Actually try to import each module and catch errors."""
    issues = {}

    for init_file in python_dir.rglob("__init__.py"):
        rel_path = init_file.parent.relative_to(PROJECT_ROOT)
        module_name = str(rel_path).replace('/', '.')

        if module_name == 'python':
            continue

        try:
            # Try to import
            exec(f"from {module_name} import *")
        except ImportError as e:
            issues[module_name] = f"ImportError: {e}"
        except Exception as e:
            issues[module_name] = f"{type(e).__name__}: {e}"

    return issues

def main():
    print("=" * 70)
    print("COMPREHENSIVE IMPORT GAP ANALYSIS")
    print("=" * 70)

    python_dir = PROJECT_ROOT / 'python'
    test_dir = PROJECT_ROOT / 'tests'

    # 1. Check test file imports
    print("\n" + "=" * 50)
    print("1. TEST FILE IMPORT ISSUES")
    print("=" * 50)
    test_issues = check_test_imports(test_dir)
    if test_issues:
        for file, issues in sorted(test_issues.items()):
            print(f"\n{file}:")
            for issue in issues:
                print(f"  - {issue}")
    else:
        print("No issues found in test imports.")

    # 2. Check __init__.py exports
    print("\n" + "=" * 50)
    print("2. MISSING EXPORTS IN __init__.py")
    print("=" * 50)
    export_issues = check_init_exports(python_dir)
    if export_issues:
        for file, missing in sorted(export_issues.items()):
            print(f"\n{file}:")
            for name in missing:
                print(f"  - Missing definition: {name}")
    else:
        print("No missing exports found.")

    # 3. Check internal imports
    print("\n" + "=" * 50)
    print("3. BROKEN INTERNAL IMPORTS")
    print("=" * 50)
    internal_issues = check_internal_imports(python_dir)
    if internal_issues:
        for file, issues in sorted(internal_issues.items()):
            print(f"\n{file}:")
            for issue in issues:
                print(f"  - {issue}")
    else:
        print("No broken internal imports found.")

    # 4. Runtime import check
    print("\n" + "=" * 50)
    print("4. RUNTIME IMPORT ERRORS")
    print("=" * 50)
    runtime_issues = runtime_import_check(python_dir)
    if runtime_issues:
        for module, error in sorted(runtime_issues.items()):
            print(f"\n{module}:")
            print(f"  {error}")
    else:
        print("All modules import successfully.")

    # Summary
    total_issues = (
        sum(len(v) for v in test_issues.values()) +
        sum(len(v) for v in export_issues.values()) +
        sum(len(v) for v in internal_issues.values()) +
        len(runtime_issues)
    )

    print("\n" + "=" * 70)
    print(f"TOTAL ISSUES FOUND: {total_issues}")
    print("=" * 70)

    return total_issues

if __name__ == "__main__":
    sys.exit(main())
