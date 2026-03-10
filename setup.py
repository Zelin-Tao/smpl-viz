from setuptools import setup, find_packages

setup(
    name="smpl-viz",
    version="0.1.0",
    description="Visualize AMASS motion capture data in MuJoCo",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "mujoco",
    ],
    entry_points={
        "console_scripts": [
            "smpl-viz=smpl_viz.cli:main",
        ],
    },
)
