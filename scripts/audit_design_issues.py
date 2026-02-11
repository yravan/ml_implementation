#!/usr/bin/env python3
"""
Comprehensive Design Audit Script

Finds design inconsistencies between test expectations and implementations:
1. Type mismatches (int vs tuple, float vs Union[int,float], etc.)
2. Missing operators on classes that tests use operators with
3. Return type mismatches
4. Missing methods/functions that tests import
5. Parameter signature mismatches
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_test_usage(test_file: str) -> Dict[str, Any]:
    """Extract how classes/functions are used in tests."""
    usage = {
        'imports': [],
        'class_usage': defaultdict(lambda: {
            'operators': set(),
            'methods_called': set(),
            'attributes_accessed': set(),
            'init_args': [],
        }),
        'function_calls': defaultdict(list),
        'assertions': []
    }

    try:
        with open(test_file) as f:
            content = f.read()
        tree = ast.parse(content)
    except:
        return usage

    # Find imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                usage['imports'].append({
                    'module': module,
                    'name': alias.name,
                    'alias': alias.asname
                })

    # Find class instantiations and method calls
    for node in ast.walk(tree):
        # Class instantiation: ClassName(args)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                args_info = []
                for arg in node.args:
                    args_info.append({'positional': True})
                for kw in node.keywords:
                    args_info.append({'keyword': kw.arg})
                usage['function_calls'][func_name].append(args_info)

        # Binary operations: a + b, a * b, a @ b, a ** b
        if isinstance(node, ast.BinOp):
            op_map = {
                ast.Add: '__add__',
                ast.Sub: '__sub__',
                ast.Mult: '__mul__',
                ast.Div: '__truediv__',
                ast.MatMult: '__matmul__',
                ast.Pow: '__pow__',
            }
            op_name = op_map.get(type(node.op))
            if op_name and isinstance(node.left, ast.Name):
                usage['class_usage'][node.left.id]['operators'].add(op_name)

        # Attribute access: obj.attr
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                usage['class_usage'][node.value.id]['attributes_accessed'].add(node.attr)

        # Assertions
        if isinstance(node, ast.Assert):
            usage['assertions'].append(ast.unparse(node) if hasattr(ast, 'unparse') else str(node))

    return usage


def extract_class_definition(source_file: str, class_name: str) -> Dict[str, Any]:
    """Extract class definition details."""
    definition = {
        'methods': {},
        'operators': set(),
        'attributes': set(),
        'init_params': [],
    }

    try:
        with open(source_file) as f:
            content = f.read()
        tree = ast.parse(content)
    except:
        return definition

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name
                    params = []
                    for arg in item.args.args:
                        if arg.arg != 'self':
                            annotation = ast.unparse(arg.annotation) if arg.annotation and hasattr(ast, 'unparse') else None
                            params.append({'name': arg.arg, 'type': annotation})

                    returns = ast.unparse(item.returns) if item.returns and hasattr(ast, 'unparse') else None
                    definition['methods'][method_name] = {
                        'params': params,
                        'returns': returns
                    }

                    if method_name.startswith('__') and method_name.endswith('__'):
                        definition['operators'].add(method_name)

                    if method_name == '__init__':
                        definition['init_params'] = params

    return definition


def extract_function_definition(source_file: str, func_name: str) -> Dict[str, Any]:
    """Extract function definition details."""
    try:
        with open(source_file) as f:
            content = f.read()
        tree = ast.parse(content)
    except:
        return {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            params = []
            for arg in node.args.args:
                annotation = ast.unparse(arg.annotation) if arg.annotation and hasattr(ast, 'unparse') else None
                params.append({'name': arg.arg, 'type': annotation})

            # Get defaults
            defaults = node.args.defaults
            num_defaults = len(defaults)
            num_params = len(params)
            for i, default in enumerate(defaults):
                param_idx = num_params - num_defaults + i
                if param_idx < len(params):
                    params[param_idx]['default'] = ast.unparse(default) if hasattr(ast, 'unparse') else str(default)

            returns = ast.unparse(node.returns) if node.returns and hasattr(ast, 'unparse') else None

            return {
                'params': params,
                'returns': returns
            }

    return {}


def find_type_annotation_issues(test_dir: str, python_dir: str) -> List[Dict]:
    """Find type annotation mismatches."""
    issues = []

    # Known patterns to check
    patterns = [
        # (test_pattern, expected_type_pattern, issue_description)
        (r'axis\s*=\s*\([\d,\s]+\)', r'axis:\s*int\b', 'axis should be Union[int, Tuple[int, ...]]'),
        (r'shape\s*=\s*\([\d,\s]+\)', r'shape:\s*int\b', 'shape should be Tuple[int, ...]'),
        (r'kernel_size\s*=\s*\([\d,\s]+\)', r'kernel_size:\s*int\b', 'kernel_size should be Union[int, Tuple[int, ...]]'),
        (r'stride\s*=\s*\([\d,\s]+\)', r'stride:\s*int\b', 'stride should be Union[int, Tuple[int, ...]]'),
        (r'padding\s*=\s*\([\d,\s]+\)', r'padding:\s*int\b', 'padding should be Union[int, Tuple[int, ...]]'),
    ]

    # Check each test file
    for test_file in Path(test_dir).glob('test_*.py'):
        with open(test_file) as f:
            test_content = f.read()

        for test_pattern, impl_pattern, issue_desc in patterns:
            if re.search(test_pattern, test_content):
                # Find corresponding implementation file
                module_name = test_file.stem.replace('test_', '')
                impl_files = list(Path(python_dir).rglob('*.py'))

                for impl_file in impl_files:
                    try:
                        with open(impl_file) as f:
                            impl_content = f.read()
                        if re.search(impl_pattern, impl_content):
                            issues.append({
                                'test_file': str(test_file),
                                'impl_file': str(impl_file),
                                'issue': issue_desc,
                                'severity': 'HIGH'
                            })
                    except:
                        pass

    return issues


def check_operator_support(test_dir: str, python_dir: str) -> List[Dict]:
    """Check if classes have required operators."""
    issues = []

    # Operators used in tests
    operators_to_check = ['__add__', '__sub__', '__mul__', '__truediv__', '__matmul__', '__pow__']

    for test_file in Path(test_dir).glob('test_*.py'):
        usage = extract_test_usage(str(test_file))

        for var_name, var_usage in usage['class_usage'].items():
            used_ops = var_usage['operators']
            if not used_ops:
                continue

            # Try to find the class definition
            for imp in usage['imports']:
                if imp['name'] == var_name or imp.get('alias') == var_name:
                    # Find source file
                    module_path = imp['module'].replace('.', '/') + '.py'
                    source_file = project_root / module_path

                    if source_file.exists():
                        class_def = extract_class_definition(str(source_file), imp['name'])
                        defined_ops = class_def.get('operators', set())

                        missing_ops = used_ops - defined_ops
                        if missing_ops:
                            issues.append({
                                'test_file': str(test_file),
                                'impl_file': str(source_file),
                                'class': imp['name'],
                                'missing_operators': list(missing_ops),
                                'issue': f"Class {imp['name']} missing operators: {missing_ops}",
                                'severity': 'CRITICAL'
                            })

    return issues


def check_return_type_matches(test_dir: str, python_dir: str) -> List[Dict]:
    """Check if return types match test expectations."""
    issues = []

    # Patterns where tests expect specific return types
    return_patterns = [
        # (test assertion pattern, expected behavior)
        (r'assert\s+(\w+)\s+is\s+True', 'should return bool'),
        (r'assert\s+(\w+)\s+is\s+False', 'should return bool'),
        (r'(\w+)\.data\b', 'should return object with .data attribute'),
        (r'(\w+)\.shape\b', 'should return object with .shape attribute'),
        (r'len\((\w+)\)', 'should return sized object'),
    ]

    for test_file in Path(test_dir).glob('test_*.py'):
        with open(test_file) as f:
            content = f.read()

        # Check for result.data patterns (expecting Variable/Tensor return)
        matches = re.findall(r'(\w+)\s*=\s*(\w+)\([^)]*\)[^\n]*\n[^\n]*\1\.data', content)
        for var_name, func_name in matches:
            issues.append({
                'test_file': str(test_file),
                'function': func_name,
                'issue': f"Test expects {func_name}() to return object with .data attribute",
                'severity': 'HIGH'
            })

    return issues


def check_context_manager_returns(python_dir: str) -> List[Dict]:
    """Check if __enter__ methods return self."""
    issues = []

    for py_file in Path(python_dir).rglob('*.py'):
        try:
            with open(py_file) as f:
                content = f.read()
            tree = ast.parse(content)
        except:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_enter = False
                enter_returns_self = False

                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__enter__':
                        has_enter = True
                        # Check if it returns self
                        for subnode in ast.walk(item):
                            if isinstance(subnode, ast.Return) and subnode.value:
                                if isinstance(subnode.value, ast.Name) and subnode.value.id == 'self':
                                    enter_returns_self = True
                                elif isinstance(subnode.value, ast.Constant) and subnode.value.value is None:
                                    pass  # Returns None

                        if has_enter and not enter_returns_self:
                            issues.append({
                                'file': str(py_file),
                                'class': node.name,
                                'issue': '__enter__ should return self for use in "with X() as y:" syntax',
                                'severity': 'MEDIUM'
                            })

    return issues


def main():
    test_dir = project_root / 'tests'
    python_dir = project_root / 'python'

    print("=" * 70)
    print("COMPREHENSIVE DESIGN AUDIT")
    print("=" * 70)

    all_issues = []

    # 1. Type annotation issues
    print("\n[1] Checking type annotation mismatches...")
    type_issues = find_type_annotation_issues(str(test_dir), str(python_dir))
    all_issues.extend(type_issues)
    for issue in type_issues:
        print(f"  - {issue['impl_file']}: {issue['issue']}")

    # 2. Missing operators
    print("\n[2] Checking missing operators on classes...")
    op_issues = check_operator_support(str(test_dir), str(python_dir))
    all_issues.extend(op_issues)
    for issue in op_issues:
        print(f"  - {issue['impl_file']}: {issue['issue']}")

    # 3. Return type mismatches
    print("\n[3] Checking return type expectations...")
    return_issues = check_return_type_matches(str(test_dir), str(python_dir))
    all_issues.extend(return_issues)
    for issue in return_issues:
        print(f"  - {issue['test_file']}: {issue['issue']}")

    # 4. Context manager issues
    print("\n[4] Checking context manager implementations...")
    ctx_issues = check_context_manager_returns(str(python_dir))
    all_issues.extend(ctx_issues)
    for issue in ctx_issues:
        print(f"  - {issue['file']}: {issue['class']} - {issue['issue']}")

    # Summary
    print("\n" + "=" * 70)
    critical = sum(1 for i in all_issues if i.get('severity') == 'CRITICAL')
    high = sum(1 for i in all_issues if i.get('severity') == 'HIGH')
    medium = sum(1 for i in all_issues if i.get('severity') == 'MEDIUM')

    print(f"TOTAL ISSUES: {len(all_issues)}")
    print(f"  CRITICAL: {critical}")
    print(f"  HIGH: {high}")
    print(f"  MEDIUM: {medium}")
    print("=" * 70)

    return len(all_issues)


if __name__ == '__main__':
    sys.exit(main())
