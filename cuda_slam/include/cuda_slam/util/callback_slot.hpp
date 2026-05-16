#pragma once

#include <algorithm>
#include <functional>
#include <mutex>
#include <vector>

namespace cuda_slam {
namespace util {

template <typename Func>
class CallbackSlot {
 public:
  using FunctionType = std::function<Func>;

  CallbackSlot() = default;
  ~CallbackSlot() = default;

  CallbackSlot(const CallbackSlot&) = delete;
  CallbackSlot& operator=(const CallbackSlot&) = delete;
  CallbackSlot(CallbackSlot&&) = default;
  CallbackSlot& operator=(CallbackSlot&&) = default;

  template <typename T>
  void add(T&& callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.emplace_back(std::forward<T>(callback));
  }

  template <typename T>
  bool remove(const T& callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(callbacks_.begin(), callbacks_.end(),
                           [&callback](const FunctionType& stored) {
                             return stored.template target<T>() ==
                                    callback.template target<T>();
                           });
    if (it != callbacks_.end()) {
      callbacks_.erase(it);
      return true;
    }
    return false;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.clear();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return callbacks_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return callbacks_.empty();
  }

  template <typename... Args>
  void call(Args&&... args) const {
    std::vector<FunctionType> snapshot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot = callbacks_;
    }
    for (const auto& callback : snapshot) {
      if (callback) {
        callback(std::forward<Args>(args)...);
      }
    }
  }

  template <typename... Args>
  void call_reverse(Args&&... args) const {
    std::vector<FunctionType> snapshot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot = callbacks_;
    }
    for (auto it = snapshot.rbegin(); it != snapshot.rend(); ++it) {
      if (*it) {
        (*it)(std::forward<Args>(args)...);
      }
    }
  }

 private:
  mutable std::mutex mutex_;
  std::vector<FunctionType> callbacks_;
};

}  // namespace util
}  // namespace cuda_slam