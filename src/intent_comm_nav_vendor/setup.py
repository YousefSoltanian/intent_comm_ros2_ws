from setuptools import setup, find_packages

package_name = "intent_comm_nav_vendor"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yousef",
    maintainer_email="yousef.soltanian75@gmail.com",
    description="Vendorized high-level controllers (human, baseline, NPACE) for navigation.",
    license="Apache-2.0",
    data_files=[
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
],

)

