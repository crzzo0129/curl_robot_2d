#pragma once
#include <algorithm>
#include <cmath>

namespace neural_controller {
// Policy ticks, not wall-clock time: a delayed update must not skip the ramp.
struct RollingHandoff {
  static constexpr double period = 0.02;
  static constexpr double candidate_delta = 0.12;
  static constexpr int blend_ticks = 15;       // 0.30 s
  static constexpr int settle_ticks = 15;      // consecutive healthy ticks
  static constexpr int settle_timeout = 100;  // 2 s after blending
  static constexpr double pi = 3.14159265358979323846;
  int ticks = 0, healthy_ticks = 0;
  int stage = 6;  // 6 blend, 7 straight settling, 8 command ramp, 3 continuous
  double alpha = 0.0, vx = 0.60, yaw = 0.0;
  bool stop_required = false;

  static bool healthy(double axis_z, double roll_rate) {
    return std::isfinite(axis_z) && std::isfinite(roll_rate) &&
        std::abs(axis_z) <= std::sin(15.0 * pi / 180.0) &&
        std::abs(roll_rate) >= 0.5 && std::abs(roll_rate) <= 12.0;
  }
  static bool window(double pitch, double axis_z, double roll_rate, double delta) {
    return healthy(axis_z, roll_rate) && std::isfinite(pitch) &&
        std::abs(pitch) <= pi / 3.0 && std::isfinite(delta) && delta <= candidate_delta;
  }
  static double slew(double previous, double target, double cap) {
    return std::clamp(target, previous - cap, previous + cap);
  }
  static double effective_action(double target, double center, double scale) {
    // Locked abduction outputs are zero by the exported actor contract.
    return scale == 0.0 ? 0.0 : std::clamp((target - center) / scale, -1.0, 1.0);
  }
  void tick(bool stable, double requested_vx, double requested_yaw) {
    ++ticks;
    if (ticks <= blend_ticks) {
      const double u = static_cast<double>(ticks) / blend_ticks;
      alpha = u * u * (3.0 - 2.0 * u);
      stage = 6;
      return;
    }
    alpha = 1.0;
    if (stage == 6 || stage == 7) {
      healthy_ticks = stable ? healthy_ticks + 1 : 0;
      stage = 7;
      if (healthy_ticks < settle_ticks) {
        stop_required = ticks >= blend_ticks + settle_timeout;
        return;
      }
      stage = 8;
    }
    vx = slew(vx, requested_vx, 0.15 * period);
    yaw = slew(yaw, requested_yaw, 0.07 * period);
    stage = std::abs(vx - requested_vx) < 1e-9 &&
        std::abs(yaw - requested_yaw) < 1e-9 ? 3 : 8;
  }
};
}  // namespace neural_controller
