import sys
import os

# Add root to sys.path
sys.path.insert(0, os.getcwd())

# Manually load .env for verification if not loaded
env_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

try:
    print("Import successful: yaml")
    print("Import successful: lida.web.app")
    print("Import successful: lida.components.gemini_generator")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback

    traceback.print_exc()
