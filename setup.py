from setuptools import setup, find_packages

setup(
    name="rf_self_training",
    version="0.1.0",
    description="Conservative self-training RF classifier for 3D point clouds with z_rel features",
    author="Your Name",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "tqdm>=4.65",
        "matplotlib>=3.7",
        "joblib>=1.3",
    ],
    extras_require={
        "shap": ["shap>=0.44"],
    },
)
