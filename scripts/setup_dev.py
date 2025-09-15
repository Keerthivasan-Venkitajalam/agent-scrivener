#!/usr/bin/env python3
"""
Development environment setup script for Agent Scrivener.
"""

import os
import sys
import subprocess
import venv
from pathlib import Path


def run_command(command, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result


def setup_virtual_environment():
    """Set up Python virtual environment."""
    project_root = Path(__file__).parent.parent
    venv_path = project_root / "venv"
    
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return venv_path
    
    print(f"Creating virtual environment at {venv_path}")
    venv.create(venv_path, with_pip=True)
    
    return venv_path


def get_python_executable(venv_path):
    """Get the Python executable path for the virtual environment."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def install_dependencies(python_exe, project_root):
    """Install project dependencies."""
    print("Installing dependencies...")
    
    # Upgrade pip
    run_command(f'"{python_exe}" -m pip install --upgrade pip')
    
    # Install requirements
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        run_command(f'"{python_exe}" -m pip install -r "{requirements_file}"')
    
    # Install project in development mode
    run_command(f'"{python_exe}" -m pip install -e .', cwd=project_root)


def setup_pre_commit_hooks(python_exe):
    """Set up pre-commit hooks for code quality."""
    print("Setting up pre-commit hooks...")
    
    try:
        run_command(f'"{python_exe}" -m pip install pre-commit')
        run_command("pre-commit install")
        print("Pre-commit hooks installed successfully")
    except subprocess.CalledProcessError:
        print("Warning: Could not set up pre-commit hooks")


def create_config_template():
    """Create a template configuration file."""
    project_root = Path(__file__).parent.parent
    config_file = project_root / "config.json"
    
    if config_file.exists():
        print("Configuration file already exists")
        return
    
    config_template = {
        "debug": True,
        "log_level": "DEBUG",
        "json_logging": False,
        "max_concurrent_sessions": 5,
        "agentcore": {
            "region": "us-east-1",
            "timeout_seconds": 300,
            "max_retries": 3
        },
        "databases": {
            "rate_limit_requests_per_minute": 30
        },
        "processing": {
            "max_sources_per_query": 10,
            "confidence_threshold": 0.7
        }
    }
    
    import json
    with open(config_file, 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"Created configuration template at {config_file}")


def run_tests(python_exe):
    """Run the test suite to verify setup."""
    print("Running tests to verify setup...")
    
    try:
        result = run_command(f'"{python_exe}" -m pytest tests/ -v', check=False)
        if result.returncode == 0:
            print("✅ All tests passed! Setup is complete.")
        else:
            print("⚠️  Some tests failed, but setup is complete.")
    except subprocess.CalledProcessError:
        print("⚠️  Could not run tests, but setup is complete.")


def main():
    """Main setup function."""
    print("🚀 Setting up Agent Scrivener development environment...")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Set up virtual environment
    venv_path = setup_virtual_environment()
    python_exe = get_python_executable(venv_path)
    
    # Install dependencies
    install_dependencies(python_exe, project_root)
    
    # Set up development tools
    setup_pre_commit_hooks(python_exe)
    
    # Create configuration template
    create_config_template()
    
    # Run tests
    run_tests(python_exe)
    
    print("\n✅ Development environment setup complete!")
    print(f"Virtual environment: {venv_path}")
    print(f"Python executable: {python_exe}")
    print("\nTo activate the virtual environment:")
    
    if sys.platform == "win32":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    
    print("\nTo run tests:")
    print("  pytest")
    
    print("\nTo start development:")
    print("  python -m agent_scrivener.api")


if __name__ == "__main__":
    main()