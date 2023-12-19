from setuptools import find_packages, setup

from chainlit_client.version import __version__

setup(
    name="chainlit_client",
    version=__version__,  # update version in chainlit_client/version.py
    description="An SDK for observability in Python applications",
    author="",
    package_data={"chainlit_client": ["py.typed"]},
    packages=find_packages(),
    install_requires=[
        "packaging>=23.0",
        "httpx>=0.23.0,<0.25.0",
        "pydantic>=1,<3",
    ],
)
