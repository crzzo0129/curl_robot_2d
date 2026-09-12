#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace neural_controller {
// Pure coordinate conversion; no inference, ROS or hardware access. Command
// channels describe the requested rolling task, not the startup policy's input.
template <class Mapping>
bool convert_rolling_history(std::array<float, 720> &destination,
                             const std::vector<float> &source,
                             const std::array<float, 12> &last_sent,
                             const Mapping &old_map, const Mapping &new_map,
                             double vx, double yaw) {
  if (source.size() != 720) return false;
  for (int frame = 0; frame < 20; ++frame) {
    const int b = 36 * frame;
    std::copy_n(source.begin() + b, 36, destination.begin() + b);
    destination[b + 6] = vx;
    destination[b + 7] = 0.0F;
    destination[b + 8] = yaw;
    destination[b + 9] = destination[b + 10] = 0.0F;
    destination[b + 11] = 1.0F;
    for (int i = 0; i < 12; ++i) {
      destination[b + 12 + i] += old_map.default_joint_pos[i] - new_map.default_joint_pos[i];
      const double target = frame == 0 ? last_sent[i] : std::clamp(
          old_map.default_joint_pos[i] + old_map.action_scales[i] * source[b + 24 + i],
          old_map.joint_lower_limits[i], old_map.joint_upper_limits[i]);
      const double scale = new_map.action_scales[i];
      const double mapped = scale == 0.0 ? 0.0 : (target - new_map.default_joint_pos[i]) / scale;
      if (!std::isfinite(mapped) || std::abs(mapped) > 1.0001) return false;
      destination[b + 24 + i] = std::clamp(mapped, -1.0, 1.0);
    }
  }
  return std::all_of(destination.begin(), destination.end(),
                     [](float value) { return std::isfinite(value); });
}
}  // namespace neural_controller
