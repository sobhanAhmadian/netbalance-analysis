import subprocess
from pathlib import Path

# Get the current script's filename
current_script = Path(__file__)
current_dir = Path(__file__).parent

# Find all Python files in the current directory
python_files = [f for f in current_dir.glob("*.py") if f != current_script]
n = len(python_files)

print(f"Running all the .py files in {current_dir}")

# Execute each Python file
for i, file in enumerate(python_files):
    print(f"\nRunning {file} [{i + 1}/{n}] ...")
    subprocess.run(["python", file], check=True)
