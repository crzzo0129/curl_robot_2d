#include "neural_controller/neural_controller.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "controller_interface/helpers.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"

namespace neural_controller {
#include "neural_controller/rolling_command_runtime.inc"

NeuralController::NeuralController()
    : controller_interface::ControllerInterface(),
      rt_cmd_vel_ptr_(nullptr),
      rt_cmd_pose_ptr_(nullptr) {}

// Check parameter vectors have the correct size
bool NeuralController::check_param_vector_size() {
  const std::vector<std::pair<std::string, size_t>> param_sizes = {
      {"action_scales", params_.action_scales.size()},
      {"action_types", params_.action_types.size()},
      {"kps", params_.kps.size()},
      {"kds", params_.kds.size()},
      {"init_kps", params_.init_kps.size()},
      {"init_kds", params_.init_kds.size()},
      {"default_joint_pos", params_.default_joint_pos.size()},
      {"joint_lower_limits", params_.joint_lower_limits.size()},
      {"joint_upper_limits", params_.joint_upper_limits.size()},
      {"joint_names", params_.joint_names.size()}};

  for (const auto &[name, size] : param_sizes) {
    if (size != kActionSize) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s size is %ld, expected %d", name.c_str(), size,
                   kActionSize);
      return false;
    }
  }
  if (!params_.startup_joint_pos.empty()) {
    if (params_.startup_joint_pos.size() != kActionSize ||
        contains_non_finite(params_.startup_joint_pos)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "startup_joint_pos must be empty or contain 12 finite values");
      return false;
    }
    for (int i = 0; i < kActionSize; ++i) {
      if (params_.startup_joint_pos.at(i) < params_.joint_lower_limits.at(i) ||
          params_.startup_joint_pos.at(i) > params_.joint_upper_limits.at(i)) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "startup_joint_pos[%d] is outside the policy joint limits", i);
        return false;
      }
    }
  }
  if (use_cem_backend_) {
    if (params_.cem_planar_joint_lower_limits.size() != 4 ||
        params_.cem_planar_joint_upper_limits.size() != 4) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "CEM planar joint limit vectors must each contain 4 values");
      return false;
    }
  }
  if (params_.roll_to_stand_enabled) {
    if (!use_cem_backend_) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "roll_to_stand_enabled requires controller_backend=cem_phase_locked");
      return false;
    }
    if (params_.roll_to_stand_stand_joint_pos.size() != kActionSize) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "roll_to_stand_stand_joint_pos must contain %d values", kActionSize);
      return false;
    }
    if (!std::isfinite(params_.roll_to_stand_trigger_pitch_deg) ||
        !std::isfinite(params_.roll_to_stand_lead_deg) ||
        !std::isfinite(params_.roll_to_stand_pitch_rate_sign) ||
        !std::isfinite(params_.roll_to_stand_min_pitch_rate) ||
        !std::isfinite(params_.roll_to_stand_min_roll_duration) ||
        !std::isfinite(params_.roll_to_stand_min_roll_turns) ||
        !std::isfinite(params_.roll_to_stand_deploy_duration)) {
      RCLCPP_ERROR(get_node()->get_logger(), "roll_to_stand parameters must be finite");
      return false;
    }
    if (params_.roll_to_stand_lead_deg <= 0.0 || params_.roll_to_stand_lead_deg >= 90.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "roll_to_stand_lead_deg must be in (0, 90)");
      return false;
    }
    if (params_.roll_to_stand_deploy_duration <= 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "roll_to_stand_deploy_duration must be positive");
      return false;
    }
    if (params_.roll_to_stand_pitch_rate_sign == 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "roll_to_stand_pitch_rate_sign must be non-zero");
      return false;
    }
    if (params_.roll_to_stand_min_pitch_rate < 0.0 ||
        params_.roll_to_stand_min_roll_duration < 0.0 ||
        params_.roll_to_stand_min_roll_turns < 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "roll_to_stand minimum values must be non-negative");
      return false;
    }
  }
  return true;
}

controller_interface::CallbackReturn NeuralController::on_init() {
  try {
    param_listener_ = std::make_shared<ParamListener>(get_node());
    params_ = param_listener_->get_params();
    if (params_.controller_backend != "rtneural" &&
        params_.controller_backend != "cem_phase_locked") {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "controller_backend must be rtneural or cem_phase_locked");
      return controller_interface::CallbackReturn::ERROR;
    }
    use_cem_backend_ = params_.controller_backend == "cem_phase_locked";

    if (params_.gain_multiplier < 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "Gain_multiplier must be >= 0.0. Stopping");
      return controller_interface::CallbackReturn::ERROR;
    }
    if (params_.gain_multiplier != 1.0) {
      RCLCPP_WARN(get_node()->get_logger(), "Gain_multiplier is set to %f",
                  params_.gain_multiplier);
    }
    if (params_.repeat_action <= 0 || params_.init_duration <= 0.0 ||
        params_.fade_in_duration < 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "repeat_action and init_duration must be positive; "
                   "fade_in_duration must be non-negative");
      return controller_interface::CallbackReturn::ERROR;
    }

    std::ifstream json_file(params_.model_path);
    if (!json_file.is_open()) {
      throw std::runtime_error("Unable to open model_path: " + params_.model_path);
    }
    nlohmann::json j;
    json_file >> j;
    command_encoding_ = parse_command_encoding(j.value("command_encoding", std::string("raw")));
    if (use_cem_backend_ && command_encoding_ != CommandEncoding::kRaw) {
      throw std::runtime_error("CEM backend requires raw command encoding");
    }
    if (j.value("requires_target_rate_limiter", false)) {
      throw std::runtime_error(
          "A rate-limited transition must use roll_to_stand_model_path, not model_path");
    }

    if (!use_cem_backend_) {
      std::ifstream json_stream(params_.model_path, std::ifstream::binary);
      model_ = RTNeural::json_parser::parseJson<float>(json_stream, true);
      if (!model_) {
        throw std::runtime_error("RTNeural failed to parse model_path: " + params_.model_path);
      }
    } else {
      model_.reset();
    }

    auto set_param_from_json_vector = [&](const std::string &key, auto &param) {
      if (j.find(key) != j.end()) {
        RCLCPP_INFO(get_node()->get_logger(), "From JSON, setting %s vector element-by-element",
                    key.c_str());
        if (j[key].size() != kActionSize) {
          std::string error_msg = "Invalid size for " + key + " (" + std::to_string(j[key].size()) +
                                  ") != " + std::to_string(kActionSize);
          RCLCPP_ERROR(get_node()->get_logger(), "%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        }
        param.resize(j[key].size(), 0.0);
        for (int i = 0; i < param.size(); i++) {
          param.at(i) = j[key].at(i);
        }
      }
    };

    auto set_param_from_json_scalar = [&](const std::string &key, auto &param, int size) {
      if (j.find(key) != j.end()) {
        RCLCPP_INFO(get_node()->get_logger(), "From JSON, setting %s[:]=%f", key.c_str(),
                    static_cast<double>(j[key]));
        param.resize(size, 0.0);
        for (auto &p : param) {
          p = j[key];
        }
      }
    };

    auto set_param_from_json_mixed = [&](const std::string &key, auto &param, int size) {
      if (j.find(key) != j.end()) {
        if (j[key].is_array()) {
          set_param_from_json_vector(key, param);
        } else {
          set_param_from_json_scalar(key, param, size);
        }
      }
    };

    set_param_from_json_scalar("kp", params_.kps, kActionSize);
    set_param_from_json_scalar("kd", params_.kds, kActionSize);
    set_param_from_json_mixed("action_scale", params_.action_scales, kActionSize);
    set_param_from_json_vector("default_joint_pos", params_.default_joint_pos);
    set_param_from_json_vector("joint_lower_limits", params_.joint_lower_limits);
    set_param_from_json_vector("joint_upper_limits", params_.joint_upper_limits);

    // Warn user that use_imu should be set in the robot description
    if (j.find("use_imu") != j.end()) {
      params_.use_imu = j["use_imu"];
      RCLCPP_WARN(get_node()->get_logger(),
                  "From JSON, setting params_use_imu=%d. Verify robot description has proper value "
                  "of use_imu too.",
                  params_.use_imu);
    }

    if (j.find("observation_history") != j.end()) {
      params_.observation_history = j["observation_history"];
      RCLCPP_INFO(get_node()->get_logger(), "From JSON, setting params_.observation_history=%ld",
                  params_.observation_history);
    }

    if (!use_cem_backend_) {
      if (j.contains("out_shape") &&
          (!j["out_shape"].is_array() || j["out_shape"].size() < 2 ||
           j["out_shape"].at(1) != kActionSize)) {
        RCLCPP_ERROR(get_node()->get_logger(), "Model out_shape must end in %d actions",
                     kActionSize);
        return controller_interface::CallbackReturn::ERROR;
      }

      // Check that the observation history is consistent with the model input shape.
      if (!j.contains("in_shape") || !j["in_shape"].is_array() ||
          j["in_shape"].size() < 2 ||
          j["in_shape"].at(1) != params_.observation_history * kSingleObservationSize) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "RTNeural in_shape must equal observation_history (%ld) * "
                     "kSingleObservationSize (%d)",
                     params_.observation_history, kSingleObservationSize);
        return controller_interface::CallbackReturn::ERROR;
      }
    } else {
      if (j.value("controller", std::string()) != "phase_locked_oscillator") {
        throw std::runtime_error(
            "CEM model must declare controller=phase_locked_oscillator");
      }

      static constexpr std::array<const char *, 8> coefficient_names = {
          "front_hip_sin",  "front_hip_cos",  "front_knee_sin", "front_knee_cos",
          "rear_hip_sin",   "rear_hip_cos",   "rear_knee_sin",  "rear_knee_cos"};
      const auto &raw_coefficients = j.at("raw_coefficients");
      for (std::size_t i = 0; i < coefficient_names.size(); ++i) {
        cem_config_.coefficients[i] = raw_coefficients.at(coefficient_names[i]).get<double>();
        if (!std::isfinite(cem_config_.coefficients[i])) {
          throw std::runtime_error("CEM coefficient is non-finite: " +
                                   std::string(coefficient_names[i]));
        }
      }
      cem_config_.oscillator_rate_rad_s = j.at("oscillator_rate_rad_s").get<double>();
      cem_config_.oscillator_coupling_per_s =
          j.at("oscillator_coupling_per_s").get<double>();
      cem_config_.knee_bias_rad = j.value("nominal_knee_bias_rad", 0.0);
      cem_config_.minimum_foot_surface_gap_m =
          j.value("minimum_foot_surface_gap_m", 0.0);
      cem_config_.foot_gap_tracking_margin_m =
          j.value("foot_gap_tracking_margin_m", 0.0);

      const std::array<std::pair<const char *, double>, 5> json_scalars = {{
          {"oscillator_rate_rad_s", cem_config_.oscillator_rate_rad_s},
          {"oscillator_coupling_per_s", cem_config_.oscillator_coupling_per_s},
          {"nominal_knee_bias_rad", cem_config_.knee_bias_rad},
          {"minimum_foot_surface_gap_m", cem_config_.minimum_foot_surface_gap_m},
          {"foot_gap_tracking_margin_m", cem_config_.foot_gap_tracking_margin_m},
      }};
      for (const auto &[name, value] : json_scalars) {
        if (!std::isfinite(value)) {
          throw std::runtime_error(std::string("CEM scalar is non-finite: ") + name);
        }
      }
      if (cem_config_.oscillator_rate_rad_s <= 0.0 ||
          cem_config_.oscillator_coupling_per_s < 0.0 ||
          cem_config_.minimum_foot_surface_gap_m < 0.0 ||
          cem_config_.foot_gap_tracking_margin_m < 0.0) {
        throw std::runtime_error("CEM rates must be valid and foot gaps must be non-negative");
      }
    }

  } catch (const std::exception &e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!check_param_vector_size()) {
    return controller_interface::CallbackReturn::ERROR;
  }
  try {
    load_neural_roll_to_stand();
    load_continuous_rolling();
  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Roll-to-stand model: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (use_cem_backend_) {
    if (!params_.use_imu) {
      RCLCPP_ERROR(get_node()->get_logger(), "The CEM phase-lock backend requires use_imu=true");
      return controller_interface::CallbackReturn::ERROR;
    }
    if (!std::isfinite(params_.cem_gyro_y_sign) || params_.cem_gyro_y_sign == 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "cem_gyro_y_sign must be finite and non-zero");
      return controller_interface::CallbackReturn::ERROR;
    }
    const std::array<double, 6> cem_runtime_scalars = {
        params_.cem_startup_ramp_duration, params_.cem_max_timestep,
        params_.cem_torso_length_m,       params_.cem_upper_link_length_m,
        params_.cem_lower_link_length_m,  params_.cem_foot_diameter_m};
    if (contains_non_finite(cem_runtime_scalars) ||
        params_.cem_startup_ramp_duration < 0.0 || params_.cem_max_timestep <= 0.0 ||
        params_.cem_torso_length_m <= 0.0 || params_.cem_upper_link_length_m <= 0.0 ||
        params_.cem_lower_link_length_m <= 0.0 || params_.cem_foot_diameter_m <= 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "Invalid CEM runtime or geometry parameter");
      return controller_interface::CallbackReturn::ERROR;
    }
    if (std::any_of(params_.action_types.begin(), params_.action_types.end(),
                    [](const std::string &type) { return type != "position"; })) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "The CEM phase-lock backend supports position actions only");
      return controller_interface::CallbackReturn::ERROR;
    }

    static constexpr std::array<int, 4> planar_indices = {1, 2, 7, 8};
    for (std::size_t i = 0; i < planar_indices.size(); ++i) {
      const int index = planar_indices[i];
      cem_config_.compact_positions[i] = params_.default_joint_pos.at(index);
      cem_config_.action_scales[i] = params_.action_scales.at(index);
      cem_config_.reference_joint_lower_limits[i] =
          params_.cem_planar_joint_lower_limits.at(i);
      cem_config_.reference_joint_upper_limits[i] =
          params_.cem_planar_joint_upper_limits.at(i);
      if (!std::isfinite(cem_config_.compact_positions[i]) ||
          !std::isfinite(cem_config_.action_scales[i]) ||
          !std::isfinite(cem_config_.reference_joint_lower_limits[i]) ||
          !std::isfinite(cem_config_.reference_joint_upper_limits[i]) ||
          cem_config_.action_scales[i] <= 0.0 ||
          cem_config_.reference_joint_lower_limits[i] >
              cem_config_.reference_joint_upper_limits[i]) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "Invalid CEM action scale or planar joint limit at index %zu", i);
        return controller_interface::CallbackReturn::ERROR;
      }
    }
    cem_config_.torso_length_m = params_.cem_torso_length_m;
    cem_config_.upper_link_length_m = params_.cem_upper_link_length_m;
    cem_config_.lower_link_length_m = params_.cem_lower_link_length_m;
    cem_config_.foot_diameter_m = params_.cem_foot_diameter_m;
    cem_config_.startup_ramp_duration_s = params_.cem_startup_ramp_duration;
    cem_config_.gyro_y_sign = params_.cem_gyro_y_sign;
    cem_controller_ = std::make_unique<CemPhaseController>(cem_config_);
    RCLCPP_INFO(get_node()->get_logger(),
                "Loaded phase-locked CEM backend from %s", params_.model_path.c_str());
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

void NeuralController::load_neural_roll_to_stand() {
  if (params_.roll_to_stand_model_path.empty()) return;
  if (use_cem_backend_ || !params_.use_imu || params_.observation_history != 20 ||
      params_.roll_to_stand_enabled ||
      std::any_of(params_.action_types.begin(), params_.action_types.end(),
                  [](const std::string &type) { return type != "position"; })) {
    throw std::runtime_error("Learned hot switching requires rtneural, IMU, 20-frame "
                             "history and position actions; disable the CEM unfolding mode");
  }
  if (!std::isfinite(params_.neural_stop_pitch_deg) ||
      !std::isfinite(params_.neural_stop_pitch_tolerance_deg) ||
      params_.neural_stop_pitch_tolerance_deg <= 0.0 ||
      params_.neural_stop_pitch_tolerance_deg >= 90.0) {
    throw std::runtime_error("Invalid neural stop pitch/window");
  }
  std::ifstream stream(params_.roll_to_stand_model_path);
  if (!stream) throw std::runtime_error("Cannot open roll_to_stand_model_path");
  nlohmann::json j;
  stream >> j;
  if (j.at("contract_version") != "transition_neural_controller_36x20_v3" ||
      j.at("required_controller_runtime") != "neural_controller_pitch_hot_switch_v1" ||
      j.at("action_semantics") != "absolute_about_default" ||
      j.at("observation_history") != 20 || j.at("single_observation_size") != 36 ||
      j.at("in_shape") != nlohmann::json::array({1, 720}) ||
      j.at("out_shape") != nlohmann::json::array({1, 12}) ||
      j.at("activation") != "elu" || j.at("actor_output") != "tanh_location" ||
      !j.at("use_imu").get<bool>() || j.at("control_orientation").get<bool>() ||
      !j.at("requires_target_rate_limiter").get<bool>() ||
      j.at("target_rate_limit_initialization") != "last_executed_roll_ctrl" ||
      j.at("target_rate_limit_order") != "joint_limits_then_rate_limit_once_per_policy_tick" ||
      j.at("transition_cmd_vel") != nlohmann::json::array({0.0, 0.0, 0.0}) ||
      j.at("desired_world_z") != nlohmann::json::array({0.0, 0.0, 1.0}) ||
      j.at("joint_names").get<std::vector<std::string>>() != params_.joint_names) {
    throw std::runtime_error("Unsupported transition contract or joint order");
  }
  transition_params_ = params_;
  auto read_vector = [&](const char *key, std::vector<double> &values) {
    values = j.at(key).get<std::vector<double>>();
    if (values.size() != kActionSize || contains_non_finite(values))
      throw std::runtime_error(std::string("Invalid transition vector: ") + key);
  };
  read_vector("action_scale", transition_params_.action_scales);
  read_vector("default_joint_pos", transition_params_.default_joint_pos);
  read_vector("joint_lower_limits", transition_params_.joint_lower_limits);
  read_vector("joint_upper_limits", transition_params_.joint_upper_limits);
  std::vector<double> rates;
  read_vector("target_rate_limits_rad_s", rates);
  transition_timestep_s_ = j.at("target_rate_limit_timestep_s").get<double>();
  const double frequency = j.at("policy_frequency_hz").get<double>();
  const double kp = j.at("kp").get<double>(), kd = j.at("kd").get<double>();
  transition_params_.observation_limit = j.at("observation_limit").get<double>();
  if (!std::isfinite(transition_timestep_s_) || transition_timestep_s_ <= 0.0 ||
      !std::isfinite(frequency) || frequency <= 0.0 ||
      std::abs(frequency * transition_timestep_s_ - 1.0) > 1e-6 ||
      !std::isfinite(kp) || !std::isfinite(kd) || kp < 0.0 || kd < 0.0 ||
      !std::isfinite(transition_params_.observation_limit) ||
      transition_params_.observation_limit <= 0.0) {
    throw std::runtime_error("Invalid transition timing, gains or observation limit");
  }
  transition_params_.kps.assign(kActionSize, kp);
  transition_params_.kds.assign(kActionSize, kd);
  for (int i = 0; i < kActionSize; ++i) {
    const double low = transition_params_.joint_lower_limits[i];
    const double high = transition_params_.joint_upper_limits[i];
    if (rates[i] <= 0.0 || transition_params_.action_scales[i] <= 0.0 || low > high ||
        transition_params_.default_joint_pos[i] < low ||
        transition_params_.default_joint_pos[i] > high) {
      throw std::runtime_error("Invalid transition action mapping or rate limit");
    }
    transition_rate_limits_[i] = rates[i];
  }
  std::ifstream model_stream(params_.roll_to_stand_model_path, std::ifstream::binary);
  transition_model_ = RTNeural::json_parser::parseJson<float>(model_stream, true);
  if (!transition_model_) throw std::runtime_error("Cannot parse transition RTNeural model");
  RCLCPP_INFO(get_node()->get_logger(), "Loaded learned roll-to-stand; trigger %.1f +/- %.1f deg",
              params_.neural_stop_pitch_deg, params_.neural_stop_pitch_tolerance_deg);
}

bool NeuralController::try_neural_roll_to_stand() {
  if (!transition_model_ || transition_active_ || !rolling_policy_ready_.load() ||
      !transition_requested_.load()) return false;
  try {
    const auto &imu = state_interfaces_map_.at(params_.imu_sensor_name);
    tf2::Quaternion q(imu.at("orientation.x").get().get_value(),
                      imu.at("orientation.y").get().get_value(),
                      imu.at("orientation.z").get().get_value(),
                      imu.at("orientation.w").get().get_value());
    if (!std::isfinite(q.length2()) || q.length2() < 1e-8)
      throw std::runtime_error("Invalid IMU quaternion while waiting for pitch gate");
    q.normalize();
    const auto row = tf2::Matrix3x3(q).getRow(2);
    // Same full-turn, nose-up-positive pitch as training snapshot selection.
    constexpr double kRadPerDeg = 3.14159265358979323846 / 180.0;
    const double pitch = std::atan2(row.x(), row.z());
    const double delta = pitch - params_.neural_stop_pitch_deg * kRadPerDeg;
    if (std::abs(std::atan2(std::sin(delta), std::cos(delta))) >
        params_.neural_stop_pitch_tolerance_deg * kRadPerDeg) return false;
    if (contains_non_finite(action_)) throw std::runtime_error("Invalid last rolling target");
    // Keep physical state and last sent targets. Cold history matches
    // reset_from_roll_state(actor_history=None), using the NEW action convention.
    std::fill(observation_.begin(), observation_.end(), 0.0F);
    for (int frame = 0; frame < 20; ++frame) {
      observation_[frame * kSingleObservationSize + kGravityZIndx] = -1.0F;
      observation_[frame * kSingleObservationSize + kDesiredWorldZIdx + 2] = 1.0F;
    }
    for (int i = 0; i < kActionSize; ++i) {
      observation_[kLastActionIdx + i] = static_cast<float>(std::clamp(
          (action_[i] - transition_params_.default_joint_pos[i]) /
              transition_params_.action_scales[i], -1.0, 1.0));
    }
    transition_requested_ = false;
    rolling_policy_ready_ = false;
    transition_active_ = true;
    continuous_active_ = false;
    continuous_requested_ = false;
    continuous_stage_ = 4;
    transition_elapsed_s_ = 0.0;
    RCLCPP_INFO(get_node()->get_logger(), "Roll-to-stand HOT SWITCH at pitch %.2f deg; no fade-in",
                pitch / kRadPerDeg);
    return true;
  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Pitch-gated switch failed: %s", e.what());
    estop_active_ = true;
    return false;
  }
}

controller_interface::CallbackReturn NeuralController::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  RCLCPP_INFO(get_node()->get_logger(), "configure successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration NeuralController::command_interface_configuration()
    const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::InterfaceConfiguration NeuralController::state_interface_configuration()
    const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::CallbackReturn NeuralController::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  // Clear command buffers to ignore pre-activation commands
  rt_cmd_vel_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Twist>>(nullptr);
  rt_cmd_pose_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Pose>>(nullptr);

  // Populate the command interfaces map
  RCLCPP_INFO(get_node()->get_logger(), "Populating command interfaces map");
  command_interfaces_map_.clear();
  for (auto &command_interface : command_interfaces_) {
    RCLCPP_INFO(get_node()->get_logger(), "Prefix %s. Adding command interface %s",
                command_interface.get_prefix_name().c_str(),
                command_interface.get_interface_name().c_str());
    command_interfaces_map_[command_interface.get_prefix_name()].insert_or_assign(
        command_interface.get_interface_name(), std::ref(command_interface));
  }

  // Populate the state interfaces map
  state_interfaces_map_.clear();
  for (auto &state_interface : state_interfaces_) {
    RCLCPP_INFO(get_node()->get_logger(), "Prefix %s. Adding state interface %s",
                state_interface.get_prefix_name().c_str(),
                state_interface.get_interface_name().c_str());
    state_interfaces_map_[state_interface.get_prefix_name()].insert_or_assign(
        state_interface.get_interface_name(), std::ref(state_interface));
  }

  // Store the initial joint positions
  for (int i = 0; i < kActionSize; i++) {
    init_joint_pos_.at(i) =
        state_interfaces_map_.at(params_.joint_names.at(i)).at("position").get().get_value();
  }

  // Reset estop caused by falling over
  transition_active_ = false;
  transition_requested_ = false;
  rolling_policy_ready_ = false;
  transition_elapsed_s_ = 0.0;
  configure_continuous_rolling_io();
  estop_active_ = false;
  policy_enable_requested_ = false;
  policy_enabled_ = !params_.require_policy_enable;
  waiting_for_policy_enable_logged_ = false;

  init_time_ = get_node()->now();
  repeat_action_counter_ = -1;
  if (cem_controller_) {
    cem_controller_->reset();
  }
  cem_has_prev_orientation_ = false;
  roll_to_stand_state_ = RollToStandState::kRolling;
  roll_to_stand_prev_in_window_ = false;
  roll_to_stand_unfold_start_s_ = 0.0;
  roll_to_stand_unfold_start_ctrl_.fill(0.0);

  cmd_x_vel_ = 0.0;
  cmd_y_vel_ = 0.0;
  cmd_yaw_vel_ = 0.0;
  desired_world_z_in_body_frame_ = tf2::Vector3(0, 0, 1);

  // Initialize the observation vector
  observation_.resize(params_.observation_history * kSingleObservationSize, 0.0);
  if (!params_.startup_joint_pos.empty() || transition_model_) {
    std::fill(observation_.begin(), observation_.end(), 0.0F);
    action_.fill(0.0F);
  }

  // Seed constant resting-orientation channels in every history frame.
  for (int i = 0; i < params_.observation_history; i++) {
    observation_.at(i * kSingleObservationSize + kGravityZIndx) = -1.0;
    observation_.at(i * kSingleObservationSize + kDesiredWorldZIdx + 2) = 1.0;
  }

  // Initialize command subscribers only for policies trained to consume them.
  if (params_.accept_external_commands) {
    cmd_vel_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
          rt_cmd_vel_ptr_.writeFromNonRT(msg);
        });

    cmd_pose_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::Pose>(
        "/cmd_pose", rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::Pose::SharedPtr msg) {
          rt_cmd_pose_ptr_.writeFromNonRT(msg);
        });
  } else {
    cmd_vel_subscriber_.reset();
    cmd_pose_subscriber_.reset();
    RCLCPP_INFO(get_node()->get_logger(),
                "External velocity and orientation commands are disabled");
  }

  policy_enable_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
      "~/enable_policy", rclcpp::SystemDefaultsQoS(),
      [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
        if (!estop_active_.load()) {
          policy_enable_requested_ = true;
        }
      });

  if (transition_model_) {
    transition_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
        "~/request_roll_to_stand", rclcpp::QoS(1).reliable().durability_volatile(),
        [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
          if (!estop_active_.load() && rolling_policy_ready_.load()) {
            continuous_requested_ = false;
            transition_requested_ = true;
            RCLCPP_INFO(get_node()->get_logger(), "Roll-to-stand armed; continuing roll until pitch gate");
          } else {
            RCLCPP_WARN(get_node()->get_logger(), "Roll-to-stand request ignored: rolling policy is not active");
          }
        });
  }

  emergency_stop_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
      "/emergency_stop", rclcpp::SystemDefaultsQoS(),
      [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
        estop_active_ = true;
        RCLCPP_INFO(get_node()->get_logger(), "Emergency stop triggered");
      });

  // emergency_stop_reset_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
  //     "/emergency_stop_reset", rclcpp::SystemDefaultsQoS(),
  //     [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
  //       if (estop_active_) {
  //         estop_active_ = false;
  //         on_activate(rclcpp_lifecycle::State());
  //         RCLCPP_INFO(get_node()->get_logger(), "Emergency stop released");
  //       }
  //     });

  // Initialize the publishers
  policy_output_publisher_ =
      get_node()->create_publisher<ActionMsg>("~/policy_output", rclcpp::SystemDefaultsQoS());
  rt_policy_output_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<ActionMsg>>(policy_output_publisher_);

  position_command_publisher_ =
      get_node()->create_publisher<ActionMsg>("~/position_command", rclcpp::SystemDefaultsQoS());
  rt_position_command_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<ActionMsg>>(position_command_publisher_);

  observation_publisher_ =
      get_node()->create_publisher<ObservationMsg>("~/observation", rclcpp::SystemDefaultsQoS());
  rt_observation_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<ObservationMsg>>(observation_publisher_);

  // Create IMU latency publishers
  imu_latency_publisher_ = get_node()->create_publisher<std_msgs::msg::Float32>(
      "~/imu_latency_seconds", rclcpp::SystemDefaultsQoS());
  rt_imu_latency_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float32>>(
          imu_latency_publisher_);

  // Create policy inference latency publishers
  policy_inference_latency_publisher_ = get_node()->create_publisher<std_msgs::msg::Float32>(
      "~/policy_inference_latency_seconds", rclcpp::SystemDefaultsQoS());
  rt_policy_inference_latency_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float32>>(
          policy_inference_latency_publisher_);

  RCLCPP_INFO(get_node()->get_logger(), "activate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn NeuralController::on_error(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  return controller_interface::CallbackReturn::FAILURE;
}

controller_interface::CallbackReturn NeuralController::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  continuous_requested_ = false;
  continuous_active_ = false;
  continuous_stage_ = 0;
  rolling_state_timer_.reset();
  rolling_request_subscriber_.reset();
  rolling_command_subscriber_.reset();
  rolling_policy_ready_ = false;
  transition_requested_ = false;
  transition_subscriber_.reset();
  rt_cmd_vel_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Twist>>(nullptr);
  rt_cmd_pose_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Pose>>(nullptr);

  for (auto &command_interface : command_interfaces_) {
    command_interface.set_value(0.0);
  }
  for (int i = 0; i < kActionSize; i++) {
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kd")
        .get()
        .set_value(params_.estop_kd);
  }

  // Clear command and state interfaces maps
  command_interfaces_map_.clear();
  state_interfaces_map_.clear();

  // Release underlying command and state interfaces
  command_interfaces_.clear();
  state_interfaces_.clear();

  // Release command and state interfaces from superclass
  release_interfaces();

  RCLCPP_INFO(get_node()->get_logger(), "Deactivate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type NeuralController::update_cem(
    const rclcpp::Time & /*time*/, const rclcpp::Duration &period, double elapsed_s) {
  double ang_vel_y = 0.0;
  double orientation_w = 0.0;
  double orientation_x = 0.0;
  double orientation_y = 0.0;
  double orientation_z = 0.0;
  double time_since_measurement_seconds = 0.0;
  std::array<double, kActionSize> joint_positions{};
  try {
    const auto &imu_interfaces = state_interfaces_map_.at(params_.imu_sensor_name);
    ang_vel_y = imu_interfaces.at("angular_velocity.y").get().get_value();
    orientation_w = imu_interfaces.at("orientation.w").get().get_value();
    orientation_x = imu_interfaces.at("orientation.x").get().get_value();
    orientation_y = imu_interfaces.at("orientation.y").get().get_value();
    orientation_z = imu_interfaces.at("orientation.z").get().get_value();
    const auto measurement_age = imu_interfaces.find("time_since_measurement_seconds");
    if (measurement_age != imu_interfaces.end()) {
      time_since_measurement_seconds = measurement_age->second.get().get_value();
    }
    for (int i = 0; i < kActionSize; ++i) {
      joint_positions.at(i) = state_interfaces_map_.at(params_.joint_names.at(i))
                                  .at("position")
                                  .get()
                                  .get_value();
    }
  } catch (const std::out_of_range &e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "CEM backend failed to read the IMU state interfaces: %s", e.what());
    return controller_interface::return_type::ERROR;
  }

  const std::array<double, 6> imu_values = {ang_vel_y,     orientation_w,
                                             orientation_x, orientation_y,
                                             orientation_z, time_since_measurement_seconds};
  if (contains_non_finite(imu_values)) {
    RCLCPP_ERROR(get_node()->get_logger(), "CEM backend received a non-finite IMU value");
    estop_active_ = true;
    return controller_interface::return_type::OK;
  }
  const double quaternion_norm_squared =
      orientation_w * orientation_w + orientation_x * orientation_x +
      orientation_y * orientation_y + orientation_z * orientation_z;
  if (quaternion_norm_squared < 1.0e-8) {
    RCLCPP_ERROR(get_node()->get_logger(), "CEM backend received an invalid IMU quaternion");
    estop_active_ = true;
    return controller_interface::return_type::OK;
  }
  if (contains_non_finite(joint_positions)) {
    RCLCPP_ERROR(get_node()->get_logger(), "CEM backend received a non-finite joint position");
    estop_active_ = true;
    return controller_interface::return_type::OK;
  }

  tf2::Quaternion q(orientation_x, orientation_y, orientation_z, orientation_w);
  q.normalize();

  // Accumulate the rolling phase from the fused IMU quaternion instead of
  // integrating the raw gyro.  Successive quaternion samples give the true
  // body rotation increment, free of gyro bias and scale-factor drift.
  double pitch_increment_rad = 0.0;
  if (cem_has_prev_orientation_) {
    tf2::Quaternion delta_q = cem_prev_orientation_.inverse() * q;
    delta_q.normalize();
    // Quaternions double-cover rotations (q and -q are the same attitude).
    // If the IMU firmware flips the quaternion sign between reports, the raw
    // product describes a ~2蟺 rotation instead of the true small increment.
    // Take the shortest path so a sign flip cannot inject a phase spike.
    if (delta_q.getW() < 0.0) {
      delta_q = tf2::Quaternion(-delta_q.getX(), -delta_q.getY(),
                                -delta_q.getZ(), -delta_q.getW());
    }
    const double vec_norm = std::sqrt(delta_q.getX() * delta_q.getX() +
                                      delta_q.getY() * delta_q.getY() +
                                      delta_q.getZ() * delta_q.getZ());
    if (vec_norm > 1.0e-12) {
      const double rotation_angle = 2.0 * std::atan2(vec_norm, delta_q.getW());
      pitch_increment_rad = rotation_angle * (delta_q.getY() / vec_norm);
    }
  }
  cem_prev_orientation_ = q;
  cem_has_prev_orientation_ = true;

  tf2::Matrix3x3 rotation(q);

  // Roll-to-stand pitch (nose-up positive, 0 = upright) and its rate.
  double pitch = 0.0;
  double pitch_rate = 0.0;
  if (params_.roll_to_stand_enabled) {
    const tf2::Vector3 row2 = rotation.getRow(2);
    pitch = std::atan2(row2.getX(), row2.getZ());
    pitch_rate = params_.roll_to_stand_pitch_rate_sign * ang_vel_y;
  }

  const tf2::Vector3 projected_gravity =
      rotation.inverse() * tf2::Vector3(0.0, 0.0, -1.0);
  if (params_.enable_body_angle_estop &&
      -projected_gravity[2] < std::cos(params_.max_body_angle)) {
    estop_active_ = true;
    RCLCPP_INFO(get_node()->get_logger(), "Emergency stop triggered");
    return controller_interface::return_type::OK;
  }

  const double raw_timestep_s = period.seconds();
  if (!std::isfinite(raw_timestep_s) || raw_timestep_s <= 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(), "Invalid CEM controller period: %f", raw_timestep_s);
    estop_active_ = true;
    return controller_interface::return_type::OK;
  }
  const double timestep_s = std::min(raw_timestep_s, params_.cem_max_timestep);
  if (raw_timestep_s > params_.cem_max_timestep) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                         "CEM controller period %.6f s exceeded limit %.6f s; clamping phase step",
                         raw_timestep_s, params_.cem_max_timestep);
  }

  const auto start_time = std::chrono::high_resolution_clock::now();
  const auto normalized_action =
      cem_controller_->update(timestep_s, pitch_increment_rad, elapsed_s);
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto inference_duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

  if (contains_non_finite(normalized_action)) {
    RCLCPP_ERROR(get_node()->get_logger(), "CEM backend produced a non-finite action");
    estop_active_ = true;
    return controller_interface::return_type::OK;
  }

  repeat_action_counter_ = (repeat_action_counter_ + 1) % params_.repeat_action;
  const bool publish_diagnostics = repeat_action_counter_ == 0;
  if (publish_diagnostics && rt_policy_output_publisher_->trylock()) {
    rt_policy_output_publisher_->msg_.data.resize(kActionSize, 0.0);
    for (int i = 0; i < kActionSize; ++i) {
      rt_policy_output_publisher_->msg_.data.at(i) =
          static_cast<float>(normalized_action.at(i));
    }
    rt_policy_output_publisher_->unlockAndPublish();
  }

  for (int i = 0; i < kActionSize; ++i) {
    const double unclipped = params_.default_joint_pos.at(i) +
                             normalized_action.at(i) * params_.action_scales.at(i);
    action_.at(i) = static_cast<float>(std::clamp(
        unclipped, params_.joint_lower_limits.at(i), params_.joint_upper_limits.at(i)));
    if (!std::isfinite(action_.at(i))) {
      RCLCPP_ERROR(get_node()->get_logger(), "CEM position command[%d] is non-finite", i);
      estop_active_ = true;
      return controller_interface::return_type::OK;
    }
  }

  // Roll-to-stand: a pitch trigger may replace the rolling targets with a
  // fast interpolation to the standing pose.
  if (params_.roll_to_stand_enabled) {
    update_roll_to_stand(pitch, pitch_rate, elapsed_s);
  }

  for (int i = 0; i < kActionSize; ++i) {
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("position")
        .get()
        .set_value(static_cast<double>(action_.at(i)));
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kp")
        .get()
        .set_value(params_.kps.at(i) * params_.gain_multiplier);
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kd")
        .get()
        .set_value(params_.kds.at(i) * params_.gain_multiplier);
  }

  if (publish_diagnostics && rt_position_command_publisher_->trylock()) {
    rt_position_command_publisher_->msg_.data.resize(kActionSize, 0.0);
    for (int i = 0; i < kActionSize; ++i) {
      rt_position_command_publisher_->msg_.data.at(i) = action_.at(i);
    }
    rt_position_command_publisher_->unlockAndPublish();
  }
  if (publish_diagnostics && rt_imu_latency_publisher_->trylock()) {
    rt_imu_latency_publisher_->msg_.data =
        static_cast<float>(time_since_measurement_seconds);
    rt_imu_latency_publisher_->unlockAndPublish();
  }
  if (publish_diagnostics && rt_policy_inference_latency_publisher_->trylock()) {
    rt_policy_inference_latency_publisher_->msg_.data =
        static_cast<float>(inference_duration_us / 1000000.0);
    rt_policy_inference_latency_publisher_->unlockAndPublish();
  }
  return controller_interface::return_type::OK;
}

controller_interface::return_type NeuralController::update(const rclcpp::Time &time,
                                                           const rclcpp::Duration &period) {
  // Emergency stop must pre-empt initialization as well as policy inference.
  if (estop_active_.load()) {
    continuous_requested_ = false;
    continuous_stage_ = 0;
    rolling_policy_ready_ = false;
    transition_requested_ = false;
    for (auto &command_interface : command_interfaces_) {
      command_interface.set_value(0.0);
    }
    for (int i = 0; i < kActionSize; i++) {
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kd")
          .get()
          .set_value(params_.estop_kd);
    }
    return controller_interface::return_type::OK;
  }

  // Startup/standby pose is independent of the learned action center.
  // Empty startup_joint_pos preserves the behavior of existing configurations.
  const auto &startup_joint_pos = params_.startup_joint_pos.empty()
                                     ? params_.default_joint_pos
                                     : params_.startup_joint_pos;
  double time_since_init = (time - init_time_).seconds();
  if (time_since_init < params_.init_duration) {
    // Follow a cubic smooth-step trajectory.  Supplying the matching desired
    // velocity avoids asking the motor-side derivative term to oppose the
    // intended initialization motion (which can otherwise cause stick-slip).
    const double phase = std::clamp(time_since_init / params_.init_duration, 0.0, 1.0);
    const double blend = phase * phase * (3.0 - 2.0 * phase);
    const double blend_velocity =
        6.0 * phase * (1.0 - phase) / params_.init_duration;

    for (int i = 0; i < kActionSize; i++) {
      const double position_delta =
          startup_joint_pos.at(i) - init_joint_pos_.at(i);
      const double interpolated_joint_pos = init_joint_pos_.at(i) + position_delta * blend;
      const double interpolated_joint_velocity = position_delta * blend_velocity;
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("position")
          .get()
          .set_value(interpolated_joint_pos);
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("velocity")
          .get()
          .set_value(interpolated_joint_velocity);
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("effort")
          .get()
          .set_value(0.0);
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kp")
          .get()
          .set_value(params_.init_kps.at(i));
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kd")
          .get()
          .set_value(params_.init_kds.at(i));
    }
    return controller_interface::return_type::OK;
  }

  // Policy outputs are position targets, so clear the initialization
  // trajectory's velocity feed-forward before any policy update or hold cycle.
  for (int i = 0; i < kActionSize; i++) {
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("velocity")
        .get()
        .set_value(0.0);
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("effort")
        .get()
        .set_value(0.0);
  }

  if (params_.require_policy_enable && !policy_enabled_) {
    if (!policy_enable_requested_.exchange(false)) {
      for (int i = 0; i < kActionSize; i++) {
        command_interfaces_map_.at(params_.joint_names.at(i))
            .at("position")
            .get()
            .set_value(startup_joint_pos.at(i));
        command_interfaces_map_.at(params_.joint_names.at(i))
            .at("kp")
            .get()
            .set_value(params_.init_kps.at(i));
        command_interfaces_map_.at(params_.joint_names.at(i))
            .at("kd")
            .get()
            .set_value(params_.init_kds.at(i));
      }
      if (!waiting_for_policy_enable_logged_) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "Holding the startup pose; publish std_msgs/Empty on ~/enable_policy to "
                    "start the configured controller backend");
        waiting_for_policy_enable_logged_ = true;
      }
      return controller_interface::return_type::OK;
    }
    policy_enabled_ = true;
    policy_enable_time_ = time;
    repeat_action_counter_ = -1;
    if (use_cem_backend_) {
      cem_controller_->reset();
      RCLCPP_WARN(get_node()->get_logger(),
                  "Controller enable received; beginning phase-locked CEM startup ramp");
    } else if (params_.fade_in_duration == 0.0) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Policy enable received; applying the first action at full scale");
    } else {
      RCLCPP_WARN(get_node()->get_logger(), "Policy enable received; beginning action fade-in");
    }
  }

  // After the init_duration has passed, fade in the policy actions
  double time_since_fade_in = params_.require_policy_enable
                                  ? (time - policy_enable_time_).seconds()
                                  : (time - init_time_).seconds() - params_.init_duration;
  if (use_cem_backend_) {
    return update_cem(time, period, time_since_fade_in);
  }
  // Check on every hardware tick, so the gate is not restricted to rolling
  // inference ticks. The first transition inference happens immediately.
  const bool just_switched = try_neural_roll_to_stand();
  if (estop_active_.load()) return controller_interface::return_type::OK;
  // A zero duration deliberately applies the first policy action at full
  // scale.  This is required by policies whose initial action creates the
  // momentum needed to enter the learned motion.
  float fade_in_multiplier =
      transition_active_ || params_.fade_in_duration == 0.0
          ? 1.0F
          : std::clamp(static_cast<float>(time_since_fade_in / params_.fade_in_duration),
                       0.0F, 1.0F);

  // Only get a new action from the policy when repeat_action_counter_ is 0
  if (transition_active_) {
    if (!just_switched) {
      const double dt = period.seconds();
      if (!std::isfinite(dt) || dt <= 0.0) {
        estop_active_ = true;
        return controller_interface::return_type::OK;
      }
      transition_elapsed_s_ += dt;
      if (transition_elapsed_s_ + 1e-9 < transition_timestep_s_)
        return controller_interface::return_type::OK;
      // Average policy frequency follows the JSON, with hardware-tick jitter.
      // Never run multiple catch-up actions after a scheduling delay.
      transition_elapsed_s_ = std::fmod(
          std::max(0.0, transition_elapsed_s_ - transition_timestep_s_), transition_timestep_s_);
    }
  } else {
    repeat_action_counter_ += 1;
    repeat_action_counter_ %= params_.repeat_action;
    if (repeat_action_counter_ != 0) return controller_interface::return_type::OK;
  }

  continuous_tick_command_ = *rolling_command_buffer_.readFromRT();
  if (continuous_active_ && (!continuous_tick_command_.valid ||
      rolling_clock() - continuous_tick_command_.received > 1.0)) {
    estop_active_ = true;
    RCLCPP_ERROR(get_node()->get_logger(), "Rolling command invalid or stale; stopping, not substituting neutral speed");
    return controller_interface::return_type::OK;
  }
  try {
    try_continuous_rolling();
  } catch (const std::exception &e) {
    continuous_requested_ = false;
    continuous_request_started_ = 0.0;
    continuous_stage_ = 5;
    RCLCPP_ERROR(get_node()->get_logger(), "Rolling takeover rejected: %s", e.what());
  }
  const auto &policy_params = transition_active_ ? transition_params_ :
      (continuous_active_ ? continuous_params_ : params_);
  const auto &policy_model = transition_active_ ? transition_model_ :
      (continuous_active_ ? continuous_model_ : model_);

  // Get the latest commanded velocities
  auto cmd_vel = rt_cmd_vel_ptr_.readFromRT();
  if (cmd_vel && cmd_vel->get()) {
    cmd_x_vel_ = cmd_vel->get()->linear.x;
    cmd_y_vel_ = cmd_vel->get()->linear.y;
    cmd_yaw_vel_ = cmd_vel->get()->angular.z;
  }

  // Get the latest commanded pose
  auto cmd_pose = rt_cmd_pose_ptr_.readFromRT();
  if (cmd_pose && cmd_pose->get()) {
    const auto &pose_msg = *cmd_pose->get();
    tf2::Quaternion q(pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z,
                      pose_msg.orientation.w);
    desired_world_z_in_body_frame_ = tf2::Vector3(0, 0, 1);
    desired_world_z_in_body_frame_ = tf2::quatRotate(q.inverse(), desired_world_z_in_body_frame_);
  }
  if (transition_active_) {
    cmd_x_vel_ = cmd_y_vel_ = cmd_yaw_vel_ = 0.0F;
    desired_world_z_in_body_frame_ = tf2::Vector3(0, 0, 1);
  } else if (continuous_active_) {
    cmd_x_vel_ = continuous_tick_command_.vx;
    cmd_y_vel_ = 0.0F;
    cmd_yaw_vel_ = continuous_tick_command_.yaw;
    desired_world_z_in_body_frame_ = tf2::Vector3(0, 0, 1);
  }

  // Get the latest observation
  double ang_vel_x = 0;
  double ang_vel_y = 0;
  double ang_vel_z = 0;
  double orientation_w = 0;
  double orientation_x = 0;
  double orientation_y = 0;
  double orientation_z = 0;
  double time_since_measurement_seconds = 0;
  try {
    // read IMU states from hardware interface
    RCLCPP_DEBUG(get_node()->get_logger(), "Attempting to read IMU angular_velocity.x from %s", params_.imu_sensor_name.c_str());
    ang_vel_x = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.x")
                    .get()
                    .get_value();
    ang_vel_y = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.y")
                    .get()
                    .get_value();
    ang_vel_z = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.z")
                    .get()
                    .get_value();
    orientation_w =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.w").get().get_value();
    orientation_x =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.x").get().get_value();
    orientation_y =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.y").get().get_value();
    orientation_z =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.z").get().get_value();

    // Try to read time_since_measurement_seconds if available (optional for simulation)
    auto imu_interfaces = state_interfaces_map_.at(params_.imu_sensor_name);
    if (imu_interfaces.find("time_since_measurement_seconds") != imu_interfaces.end()) {
      time_since_measurement_seconds = imu_interfaces.at("time_since_measurement_seconds").get().get_value();
    } else {
      // Default to 0 if not available (simulation case)
      time_since_measurement_seconds = 0.0;
      RCLCPP_DEBUG_ONCE(get_node()->get_logger(), "time_since_measurement_seconds interface not available, using default value 0.0");
    }

    // Check that the orientation is identity if we are not using the IMU. Use approximate checks
    // to avoid floating point errors
    if (!params_.use_imu) {
      if (std::abs(orientation_w - 1.0) > 1e-3 || std::abs(orientation_x) > 1e-3 ||
          std::abs(orientation_y) > 1e-3 || std::abs(orientation_z) > 1e-3) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "use_imu is false but IMU orientation is not identity");
        return controller_interface::return_type::ERROR;
      }
    } else {
      // Check that the orientation is not identity if we are using the IMU
      if (std::abs(orientation_w - 1.0) < 1e-6 && std::abs(orientation_x) < 1e-6 &&
          std::abs(orientation_y) < 1e-6 && std::abs(orientation_z) < 1e-6) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "use_imu is true but IMU orientation is near identity");
      }
    }

    // Calculate the projected gravity vector
    tf2::Quaternion q(orientation_x, orientation_y, orientation_z, orientation_w);
    if (!std::isfinite(q.length2()) || q.length2() < 1e-8) {
      estop_active_ = true;
      return controller_interface::return_type::OK;
    }
    q.normalize();
    tf2::Matrix3x3 m(q);
    tf2::Vector3 world_gravity_vector(0, 0, -1);
    tf2::Vector3 projected_gravity_vector = m.inverse() * world_gravity_vector;

    // If the maximum body angle is exceeded, trigger an emergency stop
    if (params_.enable_body_angle_estop &&
        -projected_gravity_vector[2] < cos(params_.max_body_angle)) {
      estop_active_ = true;
      RCLCPP_INFO(get_node()->get_logger(), "Emergency stop triggered");
      return controller_interface::return_type::OK;
    }

    // Fill the observation vector
    // Angular velocity
    observation_.at(0) = (float)ang_vel_x;
    observation_.at(1) = (float)ang_vel_y;
    observation_.at(2) = (float)ang_vel_z;
    // Projected gravity vector
    observation_.at(3) = (float)projected_gravity_vector[0];
    observation_.at(4) = (float)projected_gravity_vector[1];
    observation_.at(5) = (float)projected_gravity_vector[2];
    // Velocity commands
    // Transition observations keep their existing raw-command contract.
    const auto policy_command = encode_command(
        {cmd_x_vel_, cmd_y_vel_, cmd_yaw_vel_},
        (transition_active_ || continuous_active_) ? CommandEncoding::kRaw : command_encoding_);
    observation_.at(6) = policy_command[0];
    observation_.at(7) = policy_command[1];
    observation_.at(8) = policy_command[2];
    // Orientation commands
    observation_.at(kDesiredWorldZIdx) = (float)desired_world_z_in_body_frame_.getX();
    observation_.at(kDesiredWorldZIdx + 1) =
        (float)desired_world_z_in_body_frame_.getY();
    observation_.at(kDesiredWorldZIdx + 2) =
        (float)desired_world_z_in_body_frame_.getZ();

    // Joint positions
    for (int i = 0; i < kActionSize; i++) {
      // Only include the joint position in the observation if the action type
      // is position
      if (params_.action_types.at(i) == "position") {
        RCLCPP_DEBUG(get_node()->get_logger(), "Attempting to read joint position for %s (index %d)", params_.joint_names.at(i).c_str(), i);
        float joint_pos =
            state_interfaces_map_.at(params_.joint_names.at(i)).at("position").get().get_value();
        observation_.at(kJointPositionIdx + i) = joint_pos - policy_params.default_joint_pos.at(i);
      }
    }
  } catch (const std::out_of_range &e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to read states from hardware interface - std::out_of_range exception: %s", e.what());

    // Check which interfaces are missing
    RCLCPP_ERROR(get_node()->get_logger(), "=== Debug Information ===");

    // Check IMU interface
    if (state_interfaces_map_.find(params_.imu_sensor_name) == state_interfaces_map_.end()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Missing IMU sensor interface: %s", params_.imu_sensor_name.c_str());
    } else {
      RCLCPP_INFO(get_node()->get_logger(), "IMU sensor interface '%s' found", params_.imu_sensor_name.c_str());
      auto &imu_interfaces = state_interfaces_map_.at(params_.imu_sensor_name);
      std::vector<std::string> required_imu = {"angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
                                               "orientation.x", "orientation.y", "orientation.z", "orientation.w"};
      for (const auto &iface : required_imu) {
        if (imu_interfaces.find(iface) == imu_interfaces.end()) {
          RCLCPP_ERROR(get_node()->get_logger(), "Missing IMU interface: %s.%s", params_.imu_sensor_name.c_str(), iface.c_str());
        }
      }
    }

    // Check joint interfaces
    for (size_t i = 0; i < params_.joint_names.size(); i++) {
      const auto &joint_name = params_.joint_names.at(i);
      if (state_interfaces_map_.find(joint_name) == state_interfaces_map_.end()) {
        RCLCPP_ERROR(get_node()->get_logger(), "Missing joint interface: %s", joint_name.c_str());
      } else if (params_.action_types.at(i) == "position") {
        auto &joint_interfaces = state_interfaces_map_.at(joint_name);
        if (joint_interfaces.find("position") == joint_interfaces.end()) {
          RCLCPP_ERROR(get_node()->get_logger(), "Missing position interface for joint: %s", joint_name.c_str());
        }
      }
    }

    // List all available interfaces for debugging
    RCLCPP_ERROR(get_node()->get_logger(), "Available state interfaces:");
    for (const auto &[name, interfaces] : state_interfaces_map_) {
      std::string interface_list;
      for (const auto &[iface_name, iface_ref] : interfaces) {
        if (!interface_list.empty()) interface_list += ", ";
        interface_list += iface_name;
      }
      RCLCPP_ERROR(get_node()->get_logger(), "  %s: [%s]", name.c_str(), interface_list.c_str());
    }
    RCLCPP_ERROR(get_node()->get_logger(), "========================");

    return controller_interface::return_type::ERROR;
  }

  // Reject invalid sensor values before clipping so infinities cannot be
  // converted into apparently valid saturated observations.
  if (contains_non_finite(observation_)) {
    RCLCPP_ERROR(get_node()->get_logger(), "observation_ contains a non-finite value");
    estop_active_ = true;
    return controller_interface::return_type::OK;
  }

  // Clip the observation vector
  for (auto &obs : observation_) {
    obs = std::clamp(obs, static_cast<float>(-policy_params.observation_limit),
                     static_cast<float>(policy_params.observation_limit));
  }

  // Publish the observation
  if (rt_observation_publisher_->trylock()) {
    // TODO make a custom msg type with header
    // rt_observation_publisher_->msg_.header.stamp = time;
    rt_observation_publisher_->msg_.data = observation_;
    rt_observation_publisher_->unlockAndPublish();
  }

  // Measure the time before policy inference
  auto start_time = std::chrono::high_resolution_clock::now();

  // Perform policy inference
  policy_model->forward(observation_.data());

  // Measure the time after policy inference
  auto end_time = std::chrono::high_resolution_clock::now();
  auto inference_duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  RCLCPP_DEBUG(get_node()->get_logger(),
               "Policy inference took %.2f ms\tIMU measurement age: %.3f ms",
               inference_duration_us / 1000.0, time_since_measurement_seconds * 1000.0);

  // Shift the observation history to the right by kSingleObservationSize for the next control
  // step https://en.cppreference.com/w/cpp/algorithm/rotate
  std::rotate(observation_.rbegin(), observation_.rbegin() + kSingleObservationSize,
              observation_.rend());

  // Process the actions
  const float *policy_output = policy_model->getOutputs();
  for (int i = 0; i < kActionSize; i++) {
    if (!std::isfinite(policy_output[i])) {
      RCLCPP_ERROR(get_node()->get_logger(), "policy_output[%d] is non-finite", i);
      estop_active_ = true;
      return controller_interface::return_type::OK;
    }
  }

  // Publish the policy output
  if (rt_policy_output_publisher_->trylock()) {
    rt_policy_output_publisher_->msg_.data.resize(kActionSize, 0.0);
    for (int i = 0; i < kActionSize; i++) {
      rt_policy_output_publisher_->msg_.data.at(i) = policy_output[i];
    }
    // rt_policy_output_publisher_->msg_.header.stamp = time;
    rt_policy_output_publisher_->unlockAndPublish();
  }

  for (int i = 0; i < kActionSize; i++) {
    float action = policy_output[i];
    float action_scale = policy_params.action_scales.at(i);
    float default_joint_pos = policy_params.default_joint_pos.at(i);
    float lower_limit = policy_params.joint_lower_limits.at(i);
    float upper_limit = policy_params.joint_upper_limits.at(i);

    // Match the training observation contract: "last action" is the raw
    // policy output, not the fade-scaled command that is applied to hardware.
    observation_.at(kLastActionIdx + i) = action;
    // Scale and de-normalize to get the action vector
    if (params_.action_types.at(i) == "position") {
      float unclipped = fade_in_multiplier * action * action_scale + default_joint_pos;
      float target = std::clamp(unclipped, lower_limit, upper_limit);
      if (transition_active_) {
        const float max_delta = static_cast<float>(transition_rate_limits_[i] * transition_timestep_s_);
        target = std::clamp(target, action_[i] - max_delta, action_[i] + max_delta);
      }
      action_.at(i) = target;
    } else {
      action_.at(i) = fade_in_multiplier * action * action_scale;
    }

    if (!std::isfinite(action_.at(i))) {
      RCLCPP_ERROR(get_node()->get_logger(), "action_[%d] is non-finite", i);
      estop_active_ = true;
      return controller_interface::return_type::OK;
    }

    // Send the action to the hardware interface
    // Multiply by the gain multiplier to scale the gains to account for real2sim gap
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at(params_.action_types.at(i))
        .get()
        .set_value((double)action_.at(i));
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kp")
        .get()
        .set_value(policy_params.kps.at(i) * params_.gain_multiplier);
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kd")
        .get()
        .set_value(policy_params.kds.at(i) * params_.gain_multiplier);
  }

  if (!transition_active_ && fade_in_multiplier >= 1.0F) rolling_policy_ready_ = true;

  // Publish the scaled and final position command
  if (rt_position_command_publisher_->trylock()) {
    rt_position_command_publisher_->msg_.data.resize(kActionSize, 0.0);
    for (int i = 0; i < kActionSize; i++) {
      rt_position_command_publisher_->msg_.data.at(i) = action_.at(i);
    }
    // rt_position_command_publisher_->msg_.header.stamp = time;
    rt_position_command_publisher_->unlockAndPublish();
  }

  // Publish imu latency
  if (rt_imu_latency_publisher_->trylock()) {
    rt_imu_latency_publisher_->msg_.data = time_since_measurement_seconds;
    rt_imu_latency_publisher_->unlockAndPublish();
  }

  if (rt_policy_inference_latency_publisher_->trylock()) {
    rt_policy_inference_latency_publisher_->msg_.data = inference_duration_us / 1000000.0;
    rt_policy_inference_latency_publisher_->unlockAndPublish();
  }

  // Get the policy inference time
  // double policy_inference_time = (get_node()->now() - time).seconds();
  // RCLCPP_INFO(get_node()->get_logger(), "policy inference time: %f",
  // policy_inference_time);

  return controller_interface::return_type::OK;
}

void NeuralController::update_roll_to_stand(double pitch, double pitch_rate,
                                            double elapsed_s) {
  constexpr double kPi = 3.14159265358979323846;
  constexpr double kTwoPi = 2.0 * kPi;

  if (roll_to_stand_state_ == RollToStandState::kRolling) {
    const double target = params_.roll_to_stand_trigger_pitch_deg * kPi / 180.0;
    const double lead = params_.roll_to_stand_lead_deg * kPi / 180.0;
    // Shortest signed angular distance from the current pitch to the target.
    const double distance = std::atan2(std::sin(target - pitch), std::cos(target - pitch));
    const double signed_distance = distance * (pitch_rate >= 0.0 ? 1.0 : -1.0);
    const bool in_window = std::abs(pitch_rate) >= params_.roll_to_stand_min_pitch_rate &&
                           signed_distance >= 0.0 && signed_distance <= lead;
    const double accumulated_turns =
        std::abs(cem_controller_->body_phase_rad()) / kTwoPi;
    if (elapsed_s >= params_.roll_to_stand_min_roll_duration &&
        accumulated_turns >= params_.roll_to_stand_min_roll_turns &&
        in_window && !roll_to_stand_prev_in_window_) {
      roll_to_stand_state_ = RollToStandState::kUnfolding;
      roll_to_stand_unfold_start_s_ = elapsed_s;
      for (int i = 0; i < kActionSize; ++i) {
        roll_to_stand_unfold_start_ctrl_.at(i) = action_.at(i);
      }
      RCLCPP_INFO(get_node()->get_logger(),
                  "Roll-to-stand trigger at pitch %.2f deg; unfolding over %.3f s",
                  pitch * 180.0 / kPi, params_.roll_to_stand_deploy_duration);
    }
    roll_to_stand_prev_in_window_ = in_window;
  }

  if (roll_to_stand_state_ == RollToStandState::kUnfolding) {
    const double alpha = std::clamp(
        (elapsed_s - roll_to_stand_unfold_start_s_) /
            params_.roll_to_stand_deploy_duration,
        0.0, 1.0);
    for (int i = 0; i < kActionSize; ++i) {
      const double unclipped =
          (1.0 - alpha) * roll_to_stand_unfold_start_ctrl_.at(i) +
          alpha * params_.roll_to_stand_stand_joint_pos.at(i);
      action_.at(i) = static_cast<float>(std::clamp(
          unclipped, params_.joint_lower_limits.at(i), params_.joint_upper_limits.at(i)));
    }
    if (alpha >= 1.0) {
      roll_to_stand_state_ = RollToStandState::kStanding;
      RCLCPP_INFO(get_node()->get_logger(), "Roll-to-stand unfolding complete; holding stand");
    }
  }

  if (roll_to_stand_state_ == RollToStandState::kStanding) {
    for (int i = 0; i < kActionSize; ++i) {
      action_.at(i) = static_cast<float>(std::clamp(
          params_.roll_to_stand_stand_joint_pos.at(i),
          params_.joint_lower_limits.at(i), params_.joint_upper_limits.at(i)));
    }
  }
}

}  // namespace neural_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(neural_controller::NeuralController,
                       controller_interface::ControllerInterface)
