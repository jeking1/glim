#include "cuda_slam/util/config.hpp"

#include <fstream>
#include <stdexcept>

namespace cuda_slam {
namespace util {

bool Config::load(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    spdlog::error("Config: failed to open file: {}", path);
    return false;
  }

  try {
    file >> data_;
  } catch (const json::parse_error& e) {
    spdlog::error("Config: JSON parse error in file '{}': {}", path, e.what());
    return false;
  }

  spdlog::info("Config: loaded from {}", path);
  return true;
}

bool Config::load_from_string(const std::string& content) {
  try {
    data_ = json::parse(content);
  } catch (const json::parse_error& e) {
    spdlog::error("Config: JSON parse error from string: {}", e.what());
    return false;
  }
  return true;
}

void Config::save(const std::string& path) const {
  std::ofstream file(path);
  if (!file.is_open()) {
    spdlog::error("Config: failed to open file for writing: {}", path);
    throw std::runtime_error("Config: failed to open file for writing: " +
                             path);
  }
  file << std::setw(2) << data_ << std::endl;
  spdlog::info("Config: saved to {}", path);
}

std::string Config::dump(int indent) const {
  return data_.dump(indent);
}

bool Config::has_key(const std::string& key) const {
  return data_.contains(key);
}

const Config::json& Config::get_child(const std::string& key) const {
  static const json empty = json::object();
  if (!data_.contains(key)) {
    spdlog::warn("Config: key '{}' not found, returning empty object", key);
    return empty;
  }
  return data_.at(key);
}

Config Config::sub_config(const std::string& key) const {
  Config cfg;
  if (data_.contains(key)) {
    cfg.data() = data_.at(key);
  } else {
    spdlog::warn("Config: sub_config key '{}' not found, returning empty config",
                 key);
  }
  return cfg;
}

}  // namespace util
}  // namespace cuda_slam