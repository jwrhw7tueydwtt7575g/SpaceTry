from setuptools import find_packages, setup
package_name = "spaceforge_auto_trainer"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy"],
    extras_require={"hpo": ["optuna"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "AUTO_TRAINER_CONTRACT.md"]),
    ],
    zip_safe=True, tests_require=["pytest"],
)
