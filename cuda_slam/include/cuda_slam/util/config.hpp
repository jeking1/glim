#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace cuda_slam {
namespace util {

class Config {
 public:
  using json = nlohmann::json;

  Config() = default;
  ~Config() = default;

  Config(const Config&) = delete;
  Config& operator=(const Config&) = delete;
  Config(Config&&) = default;
  Config& operator=(Config&&) = default;

  bool load(const std::string& path);

  bool load_from_string(const std::string& content);

  void save(const std::string& path) const;

  std::string dump(int indent = 2) const;

  bool has_key(const std::string& key) const;

  template <typename T>
  T param(const std::string& key, const T& default_value = T{}) const {
    if (!data_.contains(key)) {
      return default_value;
    }
    try {
      return data_.at(key).get<T>();
    } catch (const json::exception& e) {
      spdlog::warn("Config: failed to parse key '{}' as requested type: {}",
                   key, e.what());
      return default_value;
    }
  }

  template <typename T>
  std::vector<T> param_vec(const std::string& key,
                           const std::vector<T>& default_value = {}) const {
    if (!data_.contains(key) || !data_.at(key).is_array()) {
      return default_value;
    }
    try {
      return data_.at(key).get<std::vector<T>>();
    } catch (const json::exception& e) {
      spdlog::warn("Config: failed to parse key '{}' as vector type: {}", key,
                   e.what());
      return default_value;
    }
  }

  std::string param_str(const std::string& key,
                        const std::string& default_value = "") const {
    return param<std::string>(key, default_value);
  }

  int param_int(const std::string& key, int default_value = 0) const {
    return param<int>(key, default_value);
  }

  double param_double(const std::string& key, double default_value = 0.0) const {
    return param<double>(key, default_value);
  }

  bool param_bool(const std::string& key, bool default_value = false) const {
    return param<bool>(key, default_value);
  }

  const json& get_child(const std::string& key) const;

  Config sub_config(const std::string& key) const;

  const json& data() const { return data_; }

  json& data() { return data_; }

 private:
  json data_;
};

}  // namespace util
}  // namespace cuda_slam