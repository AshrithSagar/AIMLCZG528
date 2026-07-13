from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ekf_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lab 4 Student',
    maintainer_email='student@example.com',
    description='Classical range-bearing EKF-SLAM demo in PyBullet + ROS 2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'landmark_sim_node = ekf_slam.pybullet_landmark_sim_node:main',
            'ekf_slam_node = ekf_slam.ekf_slam_node:main',
        ],
    },
)
