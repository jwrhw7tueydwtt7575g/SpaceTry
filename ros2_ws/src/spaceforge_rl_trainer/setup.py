from setuptools import find_packages, setup
package_name = "spaceforge_rl_trainer"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "gymnasium", "stable-baselines3", "pyyaml"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", ["config/stage1_ppo.yaml", "config/stage1_sac.yaml"]),
    ],
    zip_safe=True,
    entry_points={"console_scripts": ["spaceforge-rl-train = spaceforge_rl_trainer.cli:main"]},
)
