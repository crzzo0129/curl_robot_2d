// Included inside namespace neural_controller by neural_controller.cpp.
void NeuralController::load_continuous_rolling() {
  continuous_model_.reset();
  if (params_.rolling_model_path.empty()) return;
  if (use_cem_backend_ || !params_.use_imu || params_.observation_history != 20 ||
      params_.accept_external_commands || params_.fade_in_duration != 0.0 ||
      !params_.require_policy_enable ||
      std::any_of(params_.action_types.begin(), params_.action_types.end(),
                  [](const auto &s) { return s != "position"; }))
    throw std::runtime_error("Continuous rolling requires gated RTNeural startup, 20 frames, "
                             "IMU, position actions, no fade and no external startup commands");
  if (!std::isfinite(params_.rolling_handoff_max_target_delta_rad) ||
      params_.rolling_handoff_max_target_delta_rad <= 0.0 ||
      params_.rolling_handoff_max_target_delta_rad > 0.2)
    throw std::runtime_error("Invalid rolling target continuity threshold (0, 0.2]");
  std::ifstream stream(params_.rolling_model_path);
  if (!stream) throw std::runtime_error("Cannot open rolling_model_path");
  nlohmann::json j;
  stream >> j;
  const auto &r = j.at("controller_requirements");
  if (j.at("export_contract") != "rolling_command_ppo_36x20_batchnorm_v1" ||
      j.at("in_shape") != nlohmann::json::array({1, 720}) ||
      j.at("out_shape") != nlohmann::json::array({1, 12}) ||
      j.at("observation_history") != 20 || !j.at("use_imu").get<bool>() ||
      j.at("control_orientation").get<bool>() ||
      j.at("joint_names").get<std::vector<std::string>>() != params_.joint_names ||
      r.at("history_order") != "newest_first" ||
      r.at("command_encoding") != "raw_vx_vy_yaw_rate" ||
      r.at("policy_period_s") != 0.02 ||
      r.at("desired_world_z") != nlohmann::json::array({0, 0, 1}) ||
      r.at("last_action") != "previous_raw_tanh_action_in_rolling_action_coordinates")
    throw std::runtime_error("Unsupported rolling observation/action contract");
  const auto vx = r.at("vx_range_m_s").get<std::vector<double>>();
  const auto yaw = r.at("yaw_abs_nonzero_range_rad_s").get<std::vector<double>>();
  if (vx.size() != 2 || yaw.size() != 2 || contains_non_finite(vx) ||
      contains_non_finite(yaw) || vx[0] > 0.45 || vx[1] < 0.75 ||
      yaw[0] > 0.02 || yaw[1] < 0.07)
    throw std::runtime_error("Gamepad commands exceed the exported training range");
  continuous_params_ = params_;
  auto read = [&](const char *key, std::vector<double> &v) {
    v = j.at(key).get<std::vector<double>>();
    if (v.size() != 12 || contains_non_finite(v))
      throw std::runtime_error(std::string("Invalid rolling metadata: ") + key);
  };
  read("action_scale", continuous_params_.action_scales);
  read("default_joint_pos", continuous_params_.default_joint_pos);
  read("joint_lower_limits", continuous_params_.joint_lower_limits);
  read("joint_upper_limits", continuous_params_.joint_upper_limits);
  const double kp = j.at("kp").get<double>(), kd = j.at("kd").get<double>();
  if (!std::isfinite(kp) || !std::isfinite(kd) || kp <= 0 || kd <= 0)
    throw std::runtime_error("Invalid rolling gains");
  continuous_params_.kps.assign(12, kp);
  continuous_params_.kds.assign(12, kd);
  for (int i = 0; i < 12; ++i) {
    const double scale = i % 3 == 0 ? 0.0 : (i % 3 == 1 ? 0.8 : 1.2);
    const double center = continuous_params_.default_joint_pos[i];
    const double lo = continuous_params_.joint_lower_limits[i];
    const double hi = continuous_params_.joint_upper_limits[i];
    if (std::abs(continuous_params_.action_scales[i] - scale) > 1e-6 ||
        lo >= hi || center < lo || center > hi ||
        lo < params_.joint_lower_limits[i] - 1e-5 ||
        hi > params_.joint_upper_limits[i] + 1e-5 ||
        (i % 3 == 0 && std::abs(center - (i < 6 ? -1 : 1) *
          3.14159265358979323846 / 18.0) > 1e-6))
      throw std::runtime_error("Rolling scales, abduction center or limits mismatch");
  }
  continuous_params_.observation_limit = 100.0;
  std::ifstream weights(params_.rolling_model_path);
  continuous_model_ = RTNeural::json_parser::parseJson<float>(weights, true);
  if (!continuous_model_ || continuous_model_->getInSize() != 720 ||
      continuous_model_->getOutSize() != 12)
    throw std::runtime_error("Could not parse 720-input/12-output rolling RTNeural model");
}

void NeuralController::configure_continuous_rolling_io() {
  continuous_active_ = false;
  continuous_requested_ = false;
  continuous_stage_ = 0;
  continuous_history_ticks_ = 0;
  continuous_last_tick_ = continuous_turn_ = continuous_request_started_ = 0.0;
  rolling_command_buffer_.writeFromNonRT(RollingCommand{});
  if (!continuous_model_) return;
  rolling_command_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
      "~/rolling_cmd_vel", rclcpp::QoS(1).reliable().durability_volatile(),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        RollingCommand c;
        c.vx = msg->linear.x; c.yaw = msg->angular.z; c.received = rolling_clock();
        c.valid = std::isfinite(c.vx) && std::isfinite(c.yaw) &&
            c.vx >= 0.45 - 1e-9 && c.vx <= 0.75 + 1e-9 &&
            msg->linear.y == 0.0 && msg->linear.z == 0.0 &&
            msg->angular.x == 0.0 && msg->angular.y == 0.0 &&
            (c.yaw == 0.0 || (std::abs(c.yaw) >= 0.02 - 1e-9 && std::abs(c.yaw) <= 0.07 + 1e-9));
        rolling_command_buffer_.writeFromNonRT(c);
      });
  rolling_request_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
      "~/request_rolling_policy", rclcpp::QoS(1).reliable().durability_volatile(),
      [this](const std_msgs::msg::Empty::SharedPtr) {
        const int stage = continuous_stage_.load();
        if (!estop_active_.load() && !transition_requested_.load() &&
            (stage == 1 || stage == 5)) continuous_requested_ = true;
      });
  rolling_state_publisher_ = get_node()->create_publisher<std_msgs::msg::String>(
      "~/rolling_policy_state", rclcpp::QoS(1).reliable().durability_volatile());
  rolling_state_timer_ = get_node()->create_wall_timer(std::chrono::milliseconds(100), [this]() {
    static const char *names[] = {"inactive", "startup", "pending", "continuous", "stop", "rejected"};
    std_msgs::msg::String msg;
    msg.data = names[estop_active_.load() ? 0 : continuous_stage_.load()];
    rolling_state_publisher_->publish(msg);
  });
}

void NeuralController::try_continuous_rolling() {
  if (!continuous_model_ || continuous_active_ || transition_active_) return;
  const double now = rolling_clock();
  const auto &imu = state_interfaces_map_.at(params_.imu_sensor_name);
  tf2::Quaternion q(imu.at("orientation.x").get().get_value(),
                    imu.at("orientation.y").get().get_value(),
                    imu.at("orientation.z").get().get_value(),
                    imu.at("orientation.w").get().get_value());
  if (!std::isfinite(q.length2()) || q.length2() < 1e-8) return;
  q.normalize();
  const auto row = tf2::Matrix3x3(q).getRow(2);
  const double pitch = std::atan2(row.x(), row.z());
  const double dt = now - continuous_last_tick_;
  if (dt < 0.015 || dt > 0.025) {
    continuous_history_ticks_ = 0;
    continuous_turn_ = 0.0;
  } else {
    const double delta = pitch - continuous_pitch_;
    continuous_turn_ += std::atan2(std::sin(delta), std::cos(delta));
  }
  continuous_last_tick_ = now;
  continuous_pitch_ = pitch;
  ++continuous_history_ticks_;
  if (continuous_stage_.load() == 0) continuous_stage_ = 1;
  if (transition_requested_.load()) {
    continuous_requested_ = false;
    continuous_request_started_ = 0.0;
    continuous_stage_ = 4;
    return;
  }
  if (!continuous_requested_.load()) return;
  if (continuous_request_started_ == 0.0) continuous_request_started_ = now;
  continuous_stage_ = 2;
  if (now - continuous_request_started_ > 10.0) {
    continuous_requested_ = false;
    continuous_request_started_ = 0.0;
    continuous_stage_ = 5;
    RCLCPP_WARN(get_node()->get_logger(), "Rolling handoff rejected after 10s: timing/turn/command/target gate not met");
    return;
  }
  const auto cmd = continuous_tick_command_;
  if (!cmd.valid || now - cmd.received > 0.25 || continuous_history_ticks_ < 50 ||
      std::abs(continuous_turn_) < 2.0 * 3.14159265358979323846 ||
      std::abs(imu.at("angular_velocity.y").get().get_value()) < 0.5) return;
  // Convert every retained frame; never cold-reset a mature rolling actor.
  if (!convert_rolling_history(continuous_candidate_, observation_, action_,
                               params_, continuous_params_, cmd.vx, cmd.yaw)) return;
  // Frame zero is the current measurement; remaining frames are actual history.
  const auto gravity = tf2::Matrix3x3(q).inverse() * tf2::Vector3(0, 0, -1);
  for (int i = 0; i < 3; ++i) {
    const char *axes[] = {"angular_velocity.x", "angular_velocity.y", "angular_velocity.z"};
    continuous_candidate_[i] = imu.at(axes[i]).get().get_value();
    continuous_candidate_[3+i] = gravity[i];
  }
  for (int i = 0; i < 12; ++i)
    continuous_candidate_[12+i] = state_interfaces_map_.at(params_.joint_names[i])
        .at("position").get().get_value() - continuous_params_.default_joint_pos[i];
  if (contains_non_finite(continuous_candidate_) || contains_non_finite(action_)) return;
  for (auto &v : continuous_candidate_) v = std::clamp(v, -100.0F, 100.0F);
  continuous_model_->forward(continuous_candidate_.data());
  const auto *output = continuous_model_->getOutputs();
  for (int i = 0; i < 12; ++i) {
    if (!std::isfinite(output[i]) || std::abs(output[i]) > 1.0001 ||
        (i % 3 == 0 && output[i] != 0.0F)) return;
    const double target = std::clamp(continuous_params_.default_joint_pos[i] +
        continuous_params_.action_scales[i] * output[i],
        continuous_params_.joint_lower_limits[i], continuous_params_.joint_upper_limits[i]);
    if (std::abs(target - action_[i]) > params_.rolling_handoff_max_target_delta_rad) return;
  }
  if (estop_active_.load() || transition_requested_.load() || !continuous_requested_.load()) return;
  std::copy(continuous_candidate_.begin(), continuous_candidate_.end(), observation_.begin());
  continuous_active_ = true;
  continuous_requested_ = false;
  continuous_request_started_ = 0.0;
  continuous_stage_ = 3;
  RCLCPP_INFO(get_node()->get_logger(), "Continuous PPO rolling HOT SWITCH; live history converted, no fade");
}
