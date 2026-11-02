from setuptools import find_packages, setup
package_name = "spaceforge_mission_ctrl"
setup(name=package_name, version="0.1.0", packages=find_packages(), install_requires=["numpy"], data_files=[("share/ament_index/resource_index/packages", ["resource/" + package_name]), ("share/" + package_name, ["package.xml", "README.md", "POLICY_CONTRACT.md"])], zip_safe=True, tests_require=["pytest"], entry_points={"console_scripts": ["spaceforge-mission-ctrl = spaceforge_mission_ctrl.node:main"]})
