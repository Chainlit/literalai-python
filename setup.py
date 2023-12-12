from setuptools import find_packages, setup

setup(
    name="chainlit_python_client",
    version="0.1.0rc0",
    description="An SDK for observability in Python applications",
    author="",
    packages=find_packages(),
    install_requires=[
        "asyncio==3.4.3",
        "packaging==23.2",
        "httpx>=0.23.0,<0.25.0",
        "pydantic>=1,<3",
    ],
)
