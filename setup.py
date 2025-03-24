from setuptools import find_packages, setup

setup(
    name="literalai",
    version="0.1.201",  # update version in literalai/version.py
    description="An SDK for observability in Python applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Literal AI",
    author_email="contact@literalai.com",
    package_data={"literalai": ["py.typed"]},
    packages=find_packages(),
    license="Apache License 2.0",
    install_requires=[
        "packaging>=23.0",
        "httpx>=0.23.0",
        "pydantic>=1,<3",
        "chevron>=0.14.0",
        "traceloop-sdk>=0.33.12",
    ],
)
