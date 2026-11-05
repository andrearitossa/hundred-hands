#include "core/config.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <numbers>

namespace core {

EnvConfig load_env_config(const std::string& path) {
  const YAML::Node y = YAML::LoadFile(path);
  const auto D = [](const YAML::Node& n, const char* k) { return n[k].as<double>(); };
  const auto I = [](const YAML::Node& n, const char* k) { return n[k].as<int>(); };
  EnvConfig c{};

  const auto limits_path = std::filesystem::path(path).parent_path() / y["limits"].as<std::string>();
  const YAML::Node l = YAML::LoadFile(limits_path.string());
  c.limits = {D(l, "mass"), D(l, "v_max"), D(l, "vz_max"), D(l, "a_max"), D(l, "az_max")};

  const auto w = y["world"], b = w["buildings"], a = w["asset"];
  c.half_extent = D(w, "half_extent"); c.z_max = D(w, "z_max");
  c.bld_min = I(b, "min"); c.bld_max = I(b, "max");
  c.bld_side_min = D(b, "side_min"); c.bld_side_max = D(b, "side_max");
  c.bld_h_min = D(b, "height_min"); c.bld_h_max = D(b, "height_max");
  c.bld_asset_clear = D(b, "asset_clearance"); c.bld_spawn_clear = D(b, "spawn_clearance");
  c.asset_radius = D(a, "radius"); c.asset_half_range = D(a, "half_range"); c.asset_hit_margin = D(a, "hit_margin");

  c.drone_radius = D(y["drone"], "radius"); c.contact_distance = D(y["drone"], "contact_distance");
  const auto t = y["timing"];
  c.physics_dt = D(t, "physics_dt"); c.physics_per_control = I(t, "physics_per_control");
  c.control_per_decision = I(t, "control_per_decision"); c.max_decisions = I(t, "max_decisions");
  c.k_v = D(y["control"], "k_v");
  c.capture_radius = D(y["intercept"], "capture_radius"); c.p_n = D(y["intercept"], "p_n");

  const auto B = y["blue"];
  c.blue_count = I(B, "count"); c.blue_dist_min = D(B, "dist_min"); c.blue_dist_max = D(B, "dist_max");
  c.blue_alt_min = D(B, "alt_min"); c.blue_alt_max = D(B, "alt_max");

  const auto r = y["red"], bw = r["behaviour_weights"], wp = r["waypoints"], ev = r["evasive"];
  c.red_min = I(r, "count_min"); c.red_max = I(r, "count_max");
  c.red_alt_min = D(r, "alt_min"); c.red_alt_max = D(r, "alt_max");
  c.cruise_min = D(r, "cruise_min"); c.cruise_max = D(r, "cruise_max"); c.spawn_window = D(r, "spawn_window");
  c.behaviour_weights = {D(bw, "straight"), D(bw, "waypoint"), D(bw, "evasive")};
  c.wp_min = I(wp, "min"); c.wp_max = I(wp, "max"); c.wp_max_samples = I(wp, "max_samples");
  c.wp_lateral = D(wp, "lateral"); c.wp_reach = D(wp, "reach"); c.wp_clearance = D(wp, "clearance");
  c.ev_trigger = D(ev, "trigger"); c.ev_duration = D(ev, "duration"); c.ev_cooldown = D(ev, "cooldown");
  c.ev_max_angle = D(ev, "max_angle_deg") * std::numbers::pi / 180.0;

  const auto p = y["perception"], tr = p["tracker"], fu = p["fusion"];
  c.onboard_range = D(p, "onboard_range"); c.ground_range = D(p, "ground_range");
  c.trk_coast = D(tr, "coast"); c.trk_gate = D(tr, "gate"); c.trk_q = D(tr, "accel_var");
  c.trk_meas_var = D(tr, "meas_var"); c.trk_init_vel_var = D(tr, "init_vel_var");
  c.fus_pos = D(fu, "pos"); c.fus_vel = D(fu, "vel");
  const auto ac = y["actions"], bl = y["baseline"];
  c.act_turn = D(ac, "turn_deg") * std::numbers::pi / 180.0; c.act_climb = D(ac, "climb"); c.act_speed = D(ac, "speed");
  c.pn_gain = D(bl, "pn_gain"); c.reassign_s = D(bl, "reassign_s");
  c.guard_dist = D(bl, "guard_dist"); c.guard_alt = D(bl, "guard_alt");
  return c;
}
}  // namespace core
