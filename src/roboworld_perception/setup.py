from setuptools import find_packages, setup

package_name = "roboworld_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch",
         ["launch/perception.launch.py", "launch/isaac.launch.py"]),
        ("share/" + package_name + "/rviz", ["rviz/perception.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ingon",
    maintainer_email="157796083+ingon1026@users.noreply.github.com",
    description="Text-prompted conveyor object detection with 3D OBB pose",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = roboworld_perception.perception_node:main",
        ],
    },
)
