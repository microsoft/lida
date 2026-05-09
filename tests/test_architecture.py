import os
import ast
import pytest

PROJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "lida")


def get_imports(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except Exception:
            return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_domain_layer_independence():
    """Domain layer should not import infrastructure, application, or components."""
    domain_dir = os.path.join(PROJ_ROOT, "domain")
    for root, _, files in os.walk(domain_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imports = get_imports(file_path)
                for imp in imports:
                    assert "lida.infrastructure" not in imp
                    assert "lida.application" not in imp
                    assert "lida.components" not in imp


def test_application_layer_independence():
    """Application layer should not import infrastructure concrete classes (loosely enforced)."""
    # LidaApp might import lida.infrastructure IF it needs to instantiate something (bad practice but maybe allowed in transition).
    # But ideally LidaApp only depends on ports.
    # LidaApp.py imports lida.domain.ports
    # It should NOT import lida.infrastructure

    app_dir = os.path.join(PROJ_ROOT, "application")
    for root, _, files in os.walk(app_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imports = get_imports(file_path)
                for imp in imports:
                    if "lida.infrastructure" in imp:
                        pytest.fail(f"Application layer file {file} imports infrastructure: {imp}")
