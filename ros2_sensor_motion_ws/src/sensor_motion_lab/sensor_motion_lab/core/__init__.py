"""
ROS-agnostic core: geometry, world/map, motion & sensor models, metrics.

Nothing in this sub-package imports rclpy, so it can be unit-tested and used
in notebooks / Colab without a ROS installation.
"""

from .geometry import Angle, Pose2D, PoseArrayOps
from .metrics import (
    ConsistencyTracker,
    Direction,
    LidarSpec,
    LikelihoodSurfaceAnalyzer,
    LikelihoodSurfaceReport,
    RangeErrorAnalyzer,
    Trajectory,
    TrajectoryEvaluator,
    UMBmarkReport,
    UMBmarkRun,
)
from .motion import (
    ArcIntegrator,
    DifferentialDrive,
    DriveCalibrationError,
    DriveGeometry,
    MotionNoise,
    OdometryControl,
    OdometryMotionModel,
    VelocityControl,
    VelocityMotionModel,
)
from .sensing import (
    BeamModelParams,
    BeamRangeFinderModel,
    CorrelationModel,
    CorrelationParams,
    LikelihoodFieldModel,
    LikelihoodFieldParams,
    RangeSensorModel,
    RangeSensorModelFactory,
    Scan,
)
from .simulation import (
    AvoidanceConfig,
    ReactiveAvoidanceController,
    KinematicRobot,
    LidarConfig,
    SimulatedLidar,
    SlipNoise,
    SquarePathConfig,
    SquarePathController,
)
from .world import GridMap, GridRasterizer, RayCaster, SdfWorldWriter, WorldSpec

__all__ = [
    "Angle", "Pose2D", "PoseArrayOps",
    "ConsistencyTracker", "Direction", "LidarSpec", "LikelihoodSurfaceAnalyzer", "LikelihoodSurfaceReport",
    "RangeErrorAnalyzer",
    "Trajectory", "TrajectoryEvaluator", "UMBmarkReport", "UMBmarkRun",
    "ArcIntegrator", "DifferentialDrive", "DriveCalibrationError", "DriveGeometry", "MotionNoise",
    "OdometryControl", "OdometryMotionModel", "VelocityControl", "VelocityMotionModel",
    "BeamModelParams", "BeamRangeFinderModel", "CorrelationModel", "CorrelationParams", "LikelihoodFieldModel", "LikelihoodFieldParams", "RangeSensorModel",
    "RangeSensorModelFactory", "Scan",
    "AvoidanceConfig", "ReactiveAvoidanceController", "KinematicRobot", "LidarConfig", "SimulatedLidar", "SlipNoise", "SquarePathConfig", "SquarePathController",
    "GridMap", "GridRasterizer", "RayCaster", "SdfWorldWriter", "WorldSpec",
]
