from setuptools import find_packages, setup

setup(
    name="chainlit_client",
    version="0.1.0rc10",  # update version in chainlit_client/version.py
    description="An SDK for observability in Python applications",
    author="",
    package_data={"chainlit_client": ["py.typed"]},
    packages=find_packages(),
    install_requires=[
        "packaging>=23.0",
        "httpx>=0.23.0,<0.25.0",
        "pydantic>=1,<3",
        "nest-asyncio>=1.5.0",
    ],
)
