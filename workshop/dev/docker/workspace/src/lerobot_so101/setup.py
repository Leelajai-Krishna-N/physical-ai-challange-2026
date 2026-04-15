from setuptools import find_packages, setup

package_name = 'lerobot_so101'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, f'{package_name}.nodes'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pick_and_place_launch.py']),
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='hacker',
    maintainer_email='hacker@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_node = lerobot_so101.nodes.vision_node:main',
            'motion_planner_node = lerobot_so101.nodes.motion_planner_node:main',
        ],
    },
)
