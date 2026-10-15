from setuptools import find_packages, setup
package_name = "spaceforge_llm_planner"
setup(name=package_name, version="0.1.0", packages=find_packages(), install_requires=["numpy"], data_files=[("share/ament_index/resource_index/packages", ["resource/" + package_name]), ("share/" + package_name, ["package.xml", "README.md", "LLM_PLANNER_CONTRACT.md"])], zip_safe=True)
