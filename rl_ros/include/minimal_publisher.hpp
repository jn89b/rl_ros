#ifndef CPP_MINIMAL_PUBLISHER_H
#define CPP_MINIMAL_PUBLISHER_H

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"


class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher();
  ~MinimalPublisher();

 private:
  void timer_callback();
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};

#endif // CPP_MINIMAL_PUBLISHER_H
