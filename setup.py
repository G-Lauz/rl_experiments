import pathlib
import setuptools


def read(fname: pathlib.Path):
    directory_name = pathlib.Path(__file__).parent
    return open(directory_name / fname, "r", encoding="utf8").read()


def read_requirements(fname: pathlib.Path):
    return read(fname).strip().split("\n")


setuptools.setup(
    name="rl_experiments",
    version="0.0.1",
    author="Gabriel Lauzier",
    description="Reinforcement learning experiments",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords="reinforcement learning",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=["tests", "tests.*"]),
    install_requires=[
        *read_requirements("requirements/requirements.txt"),
        "pyro @ file://localhost/" + str(pathlib.Path("src/pyro").resolve()), # TODO: Patch, fix this
    ],
    entry_points={
        "console_scripts": [
            "ppo_cartpole=rl_experiments.ppo_cartpole:main",
            "ppo_boat=rl_experiments.ppo_boat:main",
        ]
    },
)
