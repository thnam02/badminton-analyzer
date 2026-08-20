from setuptools import setup

setup(
    name="xtcocotools",
    version="1.14.3+shim",
    description="Minimal xtcocotools shim backed by pycocotools (inference-only)",
    packages=["xtcocotools"],
    package_dir={"xtcocotools": "xtcocotools"},
    install_requires=["pycocotools"],
)
