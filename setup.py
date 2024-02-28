from setuptools import find_packages, setup

setup(
    name="literalai",
    version="0.0.208",  # update version in literalai/version.py
    description="An SDK for observability in Python applications",
    author="",
    package_data={"literalai": ["py.typed"]},
    packages=find_packages(),
    install_requires=[
        "packaging>=23.0",
        "httpx>=0.23.0",
        "pydantic>=1,<3",
    ],
)
