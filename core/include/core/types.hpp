#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace core {
using Vec3 = Eigen::Vector3d;

/// Kinematic state of an entity: position (m), velocity (m/s), world frame.
struct EntityState {
  Vec3 pos = Vec3::Zero();
  Vec3 vel = Vec3::Zero();
};

/// A sensor report. Clean in v0; covariance and confidence carry later noise and classification.
struct Detection {
  Vec3 pos;
  Eigen::Matrix3d cov;
  double p_hostile;
};

/// Tracker estimate: world frame, times in seconds. `id` is never exposed to the policy.
struct Track {
  int id = -1;
  Vec3 pos = Vec3::Zero();
  Vec3 vel = Vec3::Zero();
  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
  double age = 0;
  double since_update = 0;
};
}  // namespace core
