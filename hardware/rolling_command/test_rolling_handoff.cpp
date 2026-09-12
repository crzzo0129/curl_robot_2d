#include "neural_controller/rolling_handoff.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
using neural_controller::RollingHandoff;

int main(int argc, char**) {
  assert(RollingHandoff::window(0.0, 0.0, -5.0, 0.11));
  assert(!RollingHandoff::window(1.5, 0.0, -5.0, 0.11));
  assert(!RollingHandoff::window(0.0, 0.5, -5.0, 0.11));
  assert(!RollingHandoff::window(0.0, 0.0, -0.1, 0.11));
  assert(!RollingHandoff::window(0.0, 0.0, -5.0, 0.13));
  assert(!RollingHandoff::healthy(0, std::numeric_limits<double>::quiet_NaN()));
  assert(!RollingHandoff::healthy(std::numeric_limits<double>::infinity(), 5));
  assert(RollingHandoff::effective_action(.2, .1, 0) == 0);
  assert(std::abs(RollingHandoff::effective_action(.3, .1, .8)-.25) < 1e-12);
  RollingHandoff h;
  for (int i = 0; i < 200; ++i) {
    const double previous_vx = h.vx, previous_yaw = h.yaw, previous_alpha = h.alpha;
    const double vx = i < 90 ? .75 : .45, yaw = i < 90 ? .07 : -.07;
    h.tick(true, vx, yaw);
    assert(h.alpha >= previous_alpha && h.alpha <= 1);
    assert(std::abs(h.vx-previous_vx) <= .003+1e-12);
    assert(std::abs(h.yaw-previous_yaw) <= .0014+1e-12);
    if (i < 29) assert(h.vx == .6 && h.yaw == 0);
    if (i < 15) assert(h.stage == 6);
    if (i >= 15 && i < 29) assert(h.stage == 7);
    assert(!h.stop_required);
    if (argc > 1) std::cout << std::setprecision(17) << h.ticks << ' ' << h.stage << ' '
        << h.alpha << ' ' << h.vx << ' ' << h.yaw << '\n';
  }
  assert(h.stage == 3 && h.vx == .45 && h.yaw == -.07);
  RollingHandoff stalled;
  for (int i = 0; i < 114; ++i) stalled.tick(false, .6, .07);
  assert(!stalled.stop_required && stalled.yaw == 0);
  stalled.tick(false, .6, .07);
  assert(stalled.stop_required && stalled.yaw == 0);
  RollingHandoff interrupted;
  for (int i = 0; i < 28; ++i) interrupted.tick(true, .6, .07);
  interrupted.tick(false, .6, .07);
  for (int i = 0; i < 14; ++i) interrupted.tick(true, .6, .07);
  assert(interrupted.yaw == 0);
  interrupted.tick(true, .6, .07);
  assert(interrupted.yaw > 0);
  if (argc == 1) std::cout << "Handoff gates, blending, settling timeout, interruption and slew checks passed\n";
}
