"""
Setup script for Unified AI Analytics Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="unified-ai-platform",
    version="0.1.0",
    author="AI/ML Engineering Team",
    author_email="your.email@example.com",
    description="A comprehensive machine learning model benchmarking and analytics platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohin-io/unified-ai-platform",
    project_urls={
        "Bug Tracker": "https://github.com/mohin-io/unified-ai-platform/issues",
        "Documentation": "https://github.com/mohin-io/unified-ai-platform/docs",
        "Source Code": "https://github.com/mohin-io/unified-ai-platform",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "unified-ai=src.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning",
        "deep-learning",
        "automl",
        "mlops",
        "model-benchmarking",
        "explainable-ai",
        "data-science",
        "artificial-intelligence",
    ],
)
