from setuptools import setup, find_packages
import os

package_name = "intent_comm_hri_vendor"

setup(
    name=package_name,
    version="0.0.1",
    description="Vendors Intent-Communication-HRI python modules (iLQGame, etc.) into ROS2.",
    license="Apache-2.0",
    maintainer="you",
    maintainer_email="you@todo.com",
    # IMPORTANT: this will discover iLQGame as a top-level python package
    packages=find_packages(include=["iLQGame", "iLQGame.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)

