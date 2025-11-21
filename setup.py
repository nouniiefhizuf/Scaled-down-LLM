from setuptools import setup, find_packages

setup(
    name="llm_systems_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "pydantic",
        "pyyaml",
        "wandb",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "llm-train=scripts.train:main",
            "llm-generate=scripts.generate:main",
            "llm-benchmark=scripts.benchmark:main",
        ],
    },
    python_requires=">=3.9",
)