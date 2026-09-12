// Native inference verification only: no ROS, hardware drivers, or motor output.
#include <RTNeural/RTNeural.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc != 4) throw std::runtime_error("Usage: verify_rolling_rtneural model.json vectors.json report.json");
    std::ifstream model_file(argv[1]), vector_file(argv[2]);
    if (!model_file || !vector_file) throw std::runtime_error("Cannot open model or vectors");
    auto model = RTNeural::json_parser::parseJson<float>(model_file, true);
    if (!model || model->getInSize() != 720 || model->getOutSize() != 12)
      throw std::runtime_error("Expected a 720-input/12-output model");
    nlohmann::json vectors;
    vector_file >> vectors;
    auto inputs = vectors.at("observations").get<std::vector<std::vector<float>>>();
    auto expected = vectors.at("expected_actions").get<std::vector<std::vector<float>>>();
    if (inputs.empty() || inputs.size() != expected.size())
      throw std::runtime_error("Invalid verification sample count");
    const double tolerance = vectors.at("tolerance").get<double>();
    if (!std::isfinite(tolerance) || tolerance <= 0 || tolerance > 2e-5)
      throw std::runtime_error("Verification tolerance must be at most 2e-5");
    double maximum = 0.0, sum_squared = 0.0;
    std::vector<double> latency_ms;
    for (std::size_t n = 0; n < inputs.size(); ++n) {
      if (inputs[n].size() != 720 || expected[n].size() != 12)
        throw std::runtime_error("Malformed verification sample shape");
      for (float x : inputs[n]) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite input");
      auto start = std::chrono::steady_clock::now();
      model->forward(inputs[n].data());
      latency_ms.push_back(std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count());
      const float *actual = model->getOutputs();
      for (int i = 0; i < 12; ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[n][i]) ||
            std::abs(actual[i]) > 1.00001 || (i % 3 == 0 && actual[i] != 0.0F))
          throw std::runtime_error("Nonfinite/out-of-range output or unlocked abduction output");
        const double error = std::abs(double(actual[i]) - expected[n][i]);
        maximum = std::max(maximum, error);
        sum_squared += error * error;
      }
    }
    std::sort(latency_ms.begin(), latency_ms.end());
    nlohmann::json report = {
        {"passed", maximum <= tolerance}, {"samples", inputs.size()},
        {"max_action_error", maximum}, {"rmse", std::sqrt(sum_squared / (inputs.size() * 12))},
        {"tolerance", tolerance}, {"backend", "RTNeural Eigen float32 on robot"},
        {"inference_latency_median_ms", latency_ms[latency_ms.size() / 2]},
        {"inference_latency_max_ms", latency_ms.back()},
        {"probe_kind", "synthetic_normalizer_neighborhood"},
        {"physical_handoff_validated", false}};
    std::ofstream output(argv[3]);
    if (!output) throw std::runtime_error("Cannot write verification report");
    output << report.dump(2) << '\n';
    std::cout << report.dump(2) << '\n';
    return maximum <= tolerance ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
