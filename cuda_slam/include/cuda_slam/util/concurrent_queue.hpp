#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

namespace cuda_slam {
namespace util {

template <typename T, typename Container = std::deque<T>>
class ConcurrentQueue {
 public:
  using value_type = T;
  using container_type = Container;
  using size_type = typename Container::size_type;

  ConcurrentQueue() = default;
  ~ConcurrentQueue() = default;

  ConcurrentQueue(const ConcurrentQueue&) = delete;
  ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;
  ConcurrentQueue(ConcurrentQueue&&) = delete;
  ConcurrentQueue& operator=(ConcurrentQueue&&) = delete;

  void push(const T& item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(item);
    }
    cv_.notify_one();
  }

  void push(T&& item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(std::move(item));
    }
    cv_.notify_one();
  }

  template <typename... Args>
  void emplace(Args&&... args) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.emplace_back(std::forward<Args>(args)...);
    }
    cv_.notify_one();
  }

  bool pop(T& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  template <typename Rep, typename Period>
  bool pop_wait(T& item, const std::chrono::duration<Rep, Period>& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this]() { return !queue_.empty(); })) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  void pop_wait(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return !queue_.empty(); });
    item = std::move(queue_.front());
    queue_.pop_front();
  }

  std::vector<T> get_all_and_clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<T> result;
    result.reserve(queue_.size());
    while (!queue_.empty()) {
      result.push_back(std::move(queue_.front()));
      queue_.pop_front();
    }
    return result;
  }

  std::vector<T> get_all() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<T>(queue_.begin(), queue_.end());
  }

  size_type size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
  }

  void notify_all() { cv_.notify_all(); }

  void notify_one() { cv_.notify_one(); }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  Container queue_;
};

}  // namespace util
}  // namespace cuda_slam