from setuptools import setup, find_packages

setup(
    name="hetero-cluster-trainer",
    version="0.1.0",
    description="Performance profiler and scheduler for efficient training on heterogeneous clusters",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "ray[default]>=2.5.0",
        "pynvml>=11.5.0",
        "psutil>=5.9.0",
        "streamlit>=1.24.0",
        "plotly>=5.14.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
    entry_points={
        "console_scripts": [
            "hetero-train=src.training.main:main",
            "hetero-profile=src.profiling.main:main",
            "hetero-monitor=src.monitoring.dashboard:main",
        ],
    },
)
