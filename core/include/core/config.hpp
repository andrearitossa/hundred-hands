#pragma once
#include <array>
#include <string>

namespace core {

struct Limits { double mass, v_max, vz_max, a_max, az_max; };

struct EnvConfig {
  Limits limits;
  double half_extent, z_max;
  int bld_min, bld_max;
  double bld_side_min, bld_side_max, bld_h_min, bld_h_max, bld_asset_clear, bld_spawn_clear;
  double asset_radius, asset_half_range, asset_hit_margin;
  double drone_radius, contact_distance;
  double physics_dt;
  int physics_per_control, control_per_decision, max_decisions;
  double k_v;
  double capture_radius, p_n;
  int blue_count;
  double blue_dist_min, blue_dist_max, blue_alt_min, blue_alt_max;
  int red_min, red_max;
  double red_alt_min, red_alt_max, cruise_min, cruise_max, spawn_window;
  std::array<double, 3> behaviour_weights;  ///< straight, waypoint, evasive
  int wp_min, wp_max, wp_max_samples;
  double wp_lateral, wp_reach, wp_clearance;
  double ev_trigger, ev_duration, ev_cooldown, ev_max_angle;  ///< ev_max_angle in rad

  double onboard_range, ground_range;
  double trk_coast, trk_gate, trk_q, trk_meas_var, trk_init_vel_var, fus_pos, fus_vel;
  double act_turn, act_climb, act_speed;  ///< rad, m/s, m/s per decision
  double pn_gain, reassign_s, guard_dist, guard_alt;

  double decision_dt() const { return physics_dt * physics_per_control * control_per_decision; }
  double control_dt() const { return physics_dt * physics_per_control; }
};

/// Load an env config; `limits:` is resolved relative to the file. Throws on missing keys.
EnvConfig load_env_config(const std::string& path);
}  // namespace core
