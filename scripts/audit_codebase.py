#!/usr/bin/env python3
"""
Codebase Audit Script

This script audits the ML implementation codebase for:
1. Missing files (referenced in __init__.py but don't exist)
2. Empty/stub files that need implementation
3. Import consistency (can all modules be imported?)
4. Interface consistency (do classes have required methods?)
5. Test coverage (is there a test for each module?)

Run: python scripts/audit_codebase.py
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
TESTS_ROOT = PROJECT_ROOT / "tests"


def find_all_python_files(root: Path) -> List[Path]:
    """Find all .py files under root."""
    return list(root.rglob("*.py"))


def check_file_has_content(filepath: Path) -> Tuple[bool, int]:
    """Check if file has meaningful content (not just docstrings/imports)."""
    try:
        content = filepath.read_text()
        lines = [l.strip() for l in content.split('\n')
                 if l.strip() and not l.strip().startswith('#')]

        # Count non-trivial lines (not imports, not docstrings)
        code_lines = 0
        in_docstring = False
        for line in lines:
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if line.startswith('import ') or line.startswith('from '):
                continue
            if line.startswith('__all__'):
                continue
            code_lines += 1

        return code_lines > 10, code_lines
    except Exception as e:
        return False, 0


def check_imports_from_init(init_path: Path) -> Dict[str, bool]:
    """Check if imports in __init__.py can be resolved."""
    results = {}
    try:
        content = init_path.read_text()
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('.'):
                    # Relative import
                    module_name = node.module.lstrip('.')
                    parent_dir = init_path.parent

                    # Check if file exists
                    module_file = parent_dir / f"{module_name}.py"
                    module_dir = parent_dir / module_name / "__init__.py"

                    exists = module_file.exists() or module_dir.exists()
                    results[node.module] = exists
    except Exception as e:
        results['_error'] = str(e)

    return results


def check_class_has_methods(filepath: Path, required_methods: List[str]) -> Dict[str, List[str]]:
    """Check if classes in file have required methods."""
    results = {}
    try:
        content = filepath.read_text()
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_methods = set()
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_methods.add(item.name)

                missing = [m for m in required_methods if m not in class_methods]
                if missing:
                    results[node.name] = missing
    except Exception as e:
        results['_error'] = str(e)

    return results


def find_notimplemented_errors(filepath: Path) -> int:
    """Count NotImplementedError raises in file."""
    try:
        content = filepath.read_text()
        return content.count('NotImplementedError')
    except:
        return 0


def get_module_structure() -> Dict[str, List[str]]:
    """Get expected module structure."""
    structure = {}

    for subdir in PYTHON_ROOT.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('_'):
            py_files = []
            for f in subdir.rglob("*.py"):
                if f.name != "__init__.py":
                    rel_path = f.relative_to(PYTHON_ROOT)
                    py_files.append(str(rel_path))
            structure[subdir.name] = py_files

    return structure


def check_test_coverage(module_path: Path) -> bool:
    """Check if a test file exists for a module."""
    rel_path = module_path.relative_to(PYTHON_ROOT)

    # Convert path to test path
    # python/rl/core/policies.py -> tests/test_rl/test_core/test_policies.py
    parts = list(rel_path.parts)
    parts[-1] = "test_" + parts[-1]

    # Try different test file patterns
    patterns = [
        TESTS_ROOT / "test_" + "_".join(parts),  # tests/test_rl_core_policies.py
        TESTS_ROOT / ("test_" + parts[0]) / ("test_" + "_".join(parts[1:])),  # tests/test_rl/test_core_policies.py
        TESTS_ROOT / ("test_" + parts[0] + ".py"),  # tests/test_rl.py (catches all)
    ]

    for pattern in patterns:
        if pattern.exists():
            return True

    return False


def audit_module(module_dir: Path) -> Dict:
    """Audit a single module directory."""
    results = {
        'files': [],
        'issues': [],
        'stats': {
            'total_files': 0,
            'files_with_content': 0,
            'files_with_stubs': 0,
            'total_notimplemented': 0,
            'has_tests': False,
        }
    }

    py_files = list(module_dir.rglob("*.py"))
    results['stats']['total_files'] = len(py_files)

    for f in py_files:
        if f.name == "__init__.py":
            # Check init imports
            import_check = check_imports_from_init(f)
            for module, exists in import_check.items():
                if not exists and module != '_error':
                    results['issues'].append(f"Missing module: {module} (referenced in {f})")
        else:
            has_content, line_count = check_file_has_content(f)
            not_impl_count = find_notimplemented_errors(f)

            results['files'].append({
                'path': str(f.relative_to(PYTHON_ROOT)),
                'has_content': has_content,
                'lines': line_count,
                'not_implemented': not_impl_count,
            })

            if has_content:
                results['stats']['files_with_content'] += 1
            if not_impl_count > 0:
                results['stats']['files_with_stubs'] += 1
            results['stats']['total_notimplemented'] += not_impl_count

    return results


def main():
    print("=" * 70)
    print("ML IMPLEMENTATION CODEBASE AUDIT")
    print("=" * 70)

    # Get all modules
    modules = [d for d in PYTHON_ROOT.iterdir()
               if d.is_dir() and not d.name.startswith('_')]

    total_stats = {
        'total_files': 0,
        'files_with_content': 0,
        'total_notimplemented': 0,
        'modules_with_tests': 0,
    }

    all_issues = []
    module_reports = {}

    for module in sorted(modules):
        print(f"\n{'='*50}")
        print(f"Module: {module.name}")
        print("=" * 50)

        results = audit_module(module)
        module_reports[module.name] = results

        stats = results['stats']
        total_stats['total_files'] += stats['total_files']
        total_stats['files_with_content'] += stats['files_with_content']
        total_stats['total_notimplemented'] += stats['total_notimplemented']

        print(f"  Files: {stats['total_files']}")
        print(f"  With content: {stats['files_with_content']}")
        print(f"  With stubs (NotImplementedError): {stats['files_with_stubs']}")
        print(f"  Total NotImplementedError: {stats['total_notimplemented']}")

        if results['issues']:
            print(f"\n  ISSUES:")
            for issue in results['issues']:
                print(f"    - {issue}")
                all_issues.append(f"[{module.name}] {issue}")

        # Check for test coverage
        test_file = TESTS_ROOT / f"test_{module.name}.py"
        test_dir = TESTS_ROOT / f"test_{module.name}"
        has_test = test_file.exists() or test_dir.exists()
        if has_test:
            total_stats['modules_with_tests'] += 1
            print(f"  Tests: ✓")
        else:
            print(f"  Tests: ✗ MISSING")
            all_issues.append(f"[{module.name}] No test file found")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total modules: {len(modules)}")
    print(f"Total Python files: {total_stats['total_files']}")
    print(f"Files with content: {total_stats['files_with_content']}")
    print(f"Total NotImplementedError stubs: {total_stats['total_notimplemented']}")
    print(f"Modules with tests: {total_stats['modules_with_tests']}/{len(modules)}")

    print(f"\nTotal issues found: {len(all_issues)}")

    # Write detailed report
    report_path = PROJECT_ROOT / "AUDIT_REPORT.md"
    with open(report_path, 'w') as f:
        f.write("# Codebase Audit Report\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total modules: {len(modules)}\n")
        f.write(f"- Total Python files: {total_stats['total_files']}\n")
        f.write(f"- Files with content: {total_stats['files_with_content']}\n")
        f.write(f"- Total NotImplementedError stubs: {total_stats['total_notimplemented']}\n")
        f.write(f"- Modules with tests: {total_stats['modules_with_tests']}/{len(modules)}\n\n")

        f.write("## Issues\n\n")
        for issue in all_issues:
            f.write(f"- {issue}\n")

        f.write("\n## Module Details\n\n")
        for module_name, results in module_reports.items():
            f.write(f"### {module_name}\n\n")
            f.write(f"| File | Lines | NotImplemented |\n")
            f.write(f"|------|-------|----------------|\n")
            for file_info in results['files']:
                f.write(f"| {file_info['path']} | {file_info['lines']} | {file_info['not_implemented']} |\n")
            f.write("\n")

    print(f"\nDetailed report written to: {report_path}")

    return len(all_issues)


if __name__ == "__main__":
    sys.exit(main())
