#include "neural_controller/rolling_history.hpp"
#include <cassert>
#include <iostream>
#include <limits>

struct Mapping {
  std::vector<double> default_joint_pos = std::vector<double>(12, 0.0);
  std::vector<double> action_scales = std::vector<double>(12, 1.0);
  std::vector<double> joint_lower_limits = std::vector<double>(12, -1.0);
  std::vector<double> joint_upper_limits = std::vector<double>(12, 1.0);
};

int main() {
  Mapping old_map, next;
  next.default_joint_pos[0] = -0.1745329252;
  next.action_scales[0] = 0.0;
  next.default_joint_pos[1] = 0.1;
  next.action_scales[1] = 0.8;
  std::vector<float> history(720, 0.0F);
  for (int f = 0; f < 20; ++f) {
    history[f * 36] = static_cast<float>(f); // Detect history reordering.
    history[f * 36 + 13] = 0.4F;
    history[f * 36 + 25] = 0.5F;
  }
  const auto original = history;
  std::array<float, 12> last{};
  last[1] = 0.3F; // Actual last sent differs from the oldest frame-zero slot.
  std::array<float, 720> result{};
  assert(neural_controller::convert_rolling_history(result, history, last, old_map, next, .6, -.07));
  assert(history == original);
  for (int f = 0; f < 20; ++f) {
    const int b = f * 36;
    assert(result[b] == f);
    assert(std::abs(result[b + 12] - .1745329252) < 1e-6);
    assert(std::abs(result[b + 13] - .3) < 1e-6);
    assert(result[b + 24] == 0.0F); // Locked axes never divide by zero.
    assert(std::abs(result[b + 25] - (f == 0 ? .25 : .5)) < 1e-6);
    assert(std::abs(result[b + 6] - .6) < 1e-6 && result[b + 7] == 0);
    assert(std::abs(result[b + 8] + .07) < 1e-6 && result[b + 11] == 1);
  }
  last[1] = 1.0F; // Outside the new policy action range: do not silently clamp.
  assert(!neural_controller::convert_rolling_history(result, history, last, old_map, next, .6, 0));
  last[1] = 0.3F;
  history[40] = std::numeric_limits<float>::quiet_NaN();
  assert(!neural_controller::convert_rolling_history(result, history, last, old_map, next, .6, 0));
  std::cout << "History conversion, locked channels, ordering and rejection checks passed\n";
}
