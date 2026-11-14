#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/ros2/String_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include "robot_parameters.hpp"
#include "utils.hpp"

// Only SONIC publishes rt/lowcmd. In gamepad mode a fresh A press and the
// current request ID are required before a dance target can control the robot.
class LoopMimicDanceHandoff {
 public:
  using Command = unitree_hg::msg::dds_::LowCmd_;
  using Status = std_msgs::msg::dds_::String_;

  explicit LoopMimicDanceHandoff(bool gamepad) : gamepad_(gamepad) {
    if (gamepad_) {
      status_ = std::make_unique<unitree::robot::ChannelPublisher<Status>>("rt/loopmimic/dance_status");
      status_->InitChannel();
    }
    subscription_ =
        std::make_unique<unitree::robot::ChannelSubscriber<Command>>("rt/loopmimic/dance_target");
    subscription_->InitChannel(
        [this](const void* message) { latest_.SetData(*static_cast<const Command*>(message)); }, 1);
  }

  bool UsesGamepad() const { return gamepad_; }
  void RequestStart() { start_.store(true); }
  void RequestCancel() { cancel_.store(true); }
  bool TakePlannerRestart() { return restart_planner_.exchange(false); }
  bool BlocksMovement() const { return blocks_movement_.load(); }
  void NotifyNewPlannerMotion() { new_idle_motion_ = true; }

  bool Apply(MotorCommand& sonic, bool planner_idle) {
    using namespace std::chrono_literals;
    const auto now = Clock::now();
    const auto received = latest_.GetDataWithTime();
    const bool fresh = received.data && now - received.timestamp < 160ms;
    const bool ready = fresh && received.data->reserve()[0] == 1 && received.data->reserve()[1] == token_;
    const bool start = start_.exchange(false);
    const bool cancel = cancel_.exchange(false);

    if (phase_ == Phase::Native) {
      if (fresh && received.data->reserve()[0] == 0) armed_ = true;
      if (gamepad_ && start && !planner_idle) {
        std::cout << "[LoopMimic] Press A while standing in Planner Idle" << std::endl;
      }
      if (gamepad_ && start && planner_idle && !cancel) {
        if (++token_ == 0) ++token_;
        requested_at_ = now;
        phase_ = Phase::Requested;
        blocks_movement_ = true;
        std::cout << "[LoopMimic] Gamepad requested dance " << token_ << std::endl;
        Publish("requested");
        return false;
      }
      if (gamepad_ || !planner_idle || !ready || !armed_) {
        Publish("idle");
        return false;
      }
      BeginDance(sonic);
    }

    if (phase_ == Phase::Requested) {
      const bool declined = fresh && received.data->reserve()[1] == token_ &&
                            received.data->reserve()[0] == 0;
      if (cancel || declined || now - requested_at_ > 1200ms) {
        phase_ = Phase::Native;
        blocks_movement_ = false;
        std::cout << "[LoopMimic] Dance request cancelled before handoff" << std::endl;
        Publish("idle");
        return false;
      }
      if (!ready || !planner_idle) {
        Publish("requested");
        return false;
      }
      BeginDance(sonic);
    }

    if (phase_ == Phase::Enter || phase_ == Phase::Dance) {
      if (cancel || !ready) {
        if (fresh && received.data->reserve()[0] == 0) armed_ = true;
        origin_ = last_;
        phase_ = Phase::WaitForIdle;
        stop_sent_ = now;
        new_idle_motion_ = false;
        restart_planner_ = true;
        std::cout << "[LoopMimic] Returning to SONIC Idle" << std::endl;
      } else {
        const auto dance = Decode(*received.data);
        sonic = phase_ == Phase::Enter ? Blend(origin_, dance, ++tick_, 35) : dance;
        if (tick_ >= 35 && phase_ == Phase::Enter) {
          phase_ = Phase::Dance;
          std::cout << "[LoopMimic] Dance active" << std::endl;
        }
        last_ = sonic;
        Publish("playing");
        return true;
      }
    }

    Publish("returning");
    if (phase_ == Phase::WaitForIdle) {
      if (!new_idle_motion_) {
        if (now - stop_sent_ > 800ms) throw std::runtime_error("SONIC Idle reinitialization timed out");
        sonic = origin_;
        return true;
      }
      phase_ = Phase::Return;
      tick_ = 0;
    }

    sonic = Blend(origin_, sonic, ++tick_, 30);
    if (tick_ == 30) {
      phase_ = Phase::Native;
      blocks_movement_ = false;
      std::cout << "[LoopMimic] SONIC Idle resumed" << std::endl;
    }
    return true;
  }

 private:
  using Clock = std::chrono::steady_clock;
  enum class Phase { Native, Requested, Enter, Dance, WaitForIdle, Return };

  void BeginDance(const MotorCommand& sonic) {
    armed_ = false;
    origin_ = last_ = sonic;
    blocks_movement_ = true;
    phase_ = Phase::Enter;
    tick_ = 0;
    std::cout << "[LoopMimic] Entering dance" << std::endl;
  }

  void Publish(const char* phase) {
    if (status_) status_->Write(Status(std::string(phase) + ":" + std::to_string(token_)));
  }

  static MotorCommand Decode(const Command& input) {
    MotorCommand cmd;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      const auto& joint = input.motor_cmd()[i];
      for (float value : {joint.q(), joint.kp(), joint.kd()}) {
        if ((std::bit_cast<uint32_t>(value) & 0x7f800000) == 0x7f800000)
          throw std::runtime_error("Non-finite LoopMimic motor target");
      }
      if (std::abs(joint.q()) > 4.0f || joint.kp() < 0.0f || joint.kp() > 500.0f ||
          joint.kd() < 0.0f || joint.kd() > 20.0f)
        throw std::runtime_error("Out-of-range LoopMimic motor target");
      cmd.q_target[i] = joint.q();
      cmd.kp[i] = joint.kp();
      cmd.kd[i] = joint.kd();
    }
    return cmd;
  }

  static MotorCommand Blend(const MotorCommand& a, const MotorCommand& b, int step, int total) {
    const float t = std::min(float(step) / total, 1.0f);
    const float w = t * t * (3.0f - 2.0f * t);
    MotorCommand result;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      result.q_target[i] = std::lerp(a.q_target[i], b.q_target[i], w);
      result.dq_target[i] = std::lerp(a.dq_target[i], b.dq_target[i], w);
      result.kp[i] = std::lerp(a.kp[i], b.kp[i], w);
      result.kd[i] = std::lerp(a.kd[i], b.kd[i], w);
      result.tau_ff[i] = std::lerp(a.tau_ff[i], b.tau_ff[i], w);
    }
    return result;
  }

  const bool gamepad_;
  DataBuffer<Command> latest_;
  std::unique_ptr<unitree::robot::ChannelPublisher<Status>> status_;
  std::unique_ptr<unitree::robot::ChannelSubscriber<Command>> subscription_;
  std::atomic<bool> start_{false}, cancel_{false};
  std::atomic<bool> restart_planner_{false}, blocks_movement_{false};
  bool new_idle_motion_ = false;
  bool armed_ = true;
  Phase phase_ = Phase::Native;
  uint32_t token_ = 0;
  int tick_ = 0;
  Clock::time_point requested_at_{}, stop_sent_{};
  MotorCommand origin_{}, last_{};
};
