#!/usr/bin/env python3
"""Walking / Student / pitch-gated roll-to-stand joystick coordinator."""
import math
import time
from rolling_gamepad_mapping import rolling_command

VERSION = 'gamepad-continuous-roll-v4-ps5-8bitdo'

class Sequence:
    """ROS-independent button and settling state machine."""
    def __init__(self, indices, init_duration, hold=0.5, timeout=10.0,
                 expected_button_count=None, circle_index=1):
        if len(indices) != 5 or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError('Expected five distinct, non-negative gamepad indices')
        if expected_button_count is not None and expected_button_count <= max(indices):
            raise ValueError('Expected button count must include every configured button')
        self.indices = indices  # cross, triangle, square, estop, release
        if circle_index < 0 or circle_index in indices:
            raise ValueError('Circle must have a separate non-negative index')
        if expected_button_count is not None and circle_index >= expected_button_count:
            raise ValueError('Circle is outside the expected gamepad layout')
        self.circle_index = circle_index
        self.expected_button_count = expected_button_count
        self.previous = None
        self.state = 'idle'
        self.init_duration, self.hold, self.timeout = init_duration, hold, timeout
        self.started = 0.0
        self.stable_since = None

    def accepts_button_count(self, count):
        return count > max(self.indices + [self.circle_index]) and (
            self.expected_button_count is None or count == self.expected_button_count)

    def buttons(self, buttons):
        if not self.accepts_button_count(len(buttons)):
            self.previous = None
            return 'stop' if self.state not in ('idle', 'estop') else None
        now = [bool(buttons[i]) for i in self.indices + [self.circle_index]]
        previous = self.previous
        self.previous = now
        # E-stop acts even in the first message. Other held buttons must be
        # released after connection before a rising edge can command motion.
        if now[3]:
            return 'stop'
        if previous is None:
            return None
        edges = [a and not b for a, b in zip(now, previous)]
        if self.state == 'estop':
            return 'reset' if edges[4] else None
        if self.state == 'switching':
            return None
        if edges[2] and self.state in ('rolling', 'roll_requested', 'continuous_rolling'):
            return 'stand'
        if edges[5] and self.state == 'rolling':
            return 'continuous_roll'
        if edges[1] and self.state == 'walking':
            return 'roll'
        if edges[0] and self.state == 'idle':
            return 'walk'
        if edges[0] and self.state in ('stand_requested', 'walk_return_blocked'):
            return 'return_walk'
        return None

    def request_walk_return(self, now):
        self.state = 'waiting_walk'
        self.started, self.stable_since = now, None

    def walk_return(self, now, stable):
        if self.state != 'waiting_walk':
            return None
        if now-self.started > self.timeout:
            return 'cancel'
        if not stable:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = now
        elif now-self.stable_since >= self.hold:
            return 'walk'
        return None

    def switched(self, target, now):
        self.state = 'walking' if target == 'walk' else 'settling'
        self.started, self.stable_since = now, None

    def settle(self, now, stable):
        if self.state != 'settling':
            return None
        if now-self.started > self.timeout:
            return 'stop'
        if not stable or now-self.started < self.init_duration:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = now
        elif now-self.stable_since >= self.hold:
            return 'enable'
        return None


def main():
    import rclpy
    from rclpy.node import Node
    from controller_manager_msgs.srv import ListControllers, SwitchController
    from sensor_msgs.msg import Joy, JointState, Imu
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Empty, Float32MultiArray, String

    class Coordinator(Node):
        def __init__(self):
            super().__init__('joy_util_node')
            # PS5 through the 8BitDo receiver's Xbox/XInput output.
            defaults = dict(cross_index=0, triangle_index=3, square_index=2,
                            estop_index=10, estop_release_index=7,
                            expected_button_count=11, circle_index=1,
                            rolling_forward_axis=1, rolling_yaw_axis=3,
                            rolling_joystick_deadzone=0.10,
                            walking_controller='neural_controller',
                            rolling_controller='neural_controller_roll',
                            startup_joint_pos=[0., .9, 1.15]*4, joint_names=[''],
                            init_duration=1.0, stand_hold_seconds=0.5,
                            stand_timeout_seconds=10.0, joint_tolerance=0.10,
                            velocity_tolerance=0.25, gyro_tolerance=0.3, stand_tilt_tolerance=0.35,
                            joy_timeout_seconds=1.0)
            for key, value in defaults.items():
                self.declare_parameter(key, value)
            self.p = {key:self.get_parameter(key).value for key in defaults}
            self.seq = Sequence([self.p[k] for k in ('cross_index','triangle_index',
                                'square_index','estop_index','estop_release_index')],
                                self.p['init_duration'], self.p['stand_hold_seconds'],
                                self.p['stand_timeout_seconds'],
                                expected_button_count=self.p['expected_button_count'],
                                circle_index=self.p['circle_index'])
            self.rolling_axes = None
            self.rolling_stage = 'unavailable'
            self.rolling_stage_time = -math.inf
            self.rolling_pending_seen = False
            self.bad_button_count = None
            if len(self.p['joint_names']) != 12 or len(self.p['startup_joint_pos']) != 12:
                raise ValueError('Expected 12 joint names and startup positions')
            self.walk, self.roll = self.p['walking_controller'], self.p['rolling_controller']
            self.controllers = {}
            self.controllers_time = -math.inf
            self.blocked_reason = ''
            self.list_future = self.pending = None
            self.generation = 0
            self.joy_time = self.joint_time = self.imu_time = -math.inf
            self.joints = self.imu = None
            self.enable_time = None
            self.estop = self.create_publisher(Empty, '/emergency_stop', 10)
            self.enable = self.create_publisher(Empty, f'/{self.roll}/enable_policy', 10)
            self.stand = self.create_publisher(Empty, f'/{self.roll}/request_roll_to_stand', 10)
            self.continuous = self.create_publisher(Empty, f'/{self.roll}/request_rolling_policy', 1)
            self.rolling_velocity = self.create_publisher(Twist, f'/{self.roll}/rolling_cmd_vel', 1)
            self.status = self.create_publisher(String, '~/sequence_state', 10)
            self.detail = self.create_publisher(String, '~/sequence_detail', 10)
            self.create_subscription(Joy, '/joy', self.joy, 10)
            self.create_subscription(String, f'/{self.roll}/rolling_policy_state', self.policy_state, 1)
            self.create_subscription(JointState, '/joint_states', self.joint, 10)
            self.create_subscription(Imu, '/imu_sensor_broadcaster/imu', self.inertial, 10)
            self.create_subscription(Float32MultiArray, f'/{self.roll}/policy_output', self.output, 10)
            self.create_subscription(Empty, '/emergency_stop', lambda _:self.latch_stop(), 10)
            self.list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
            self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
            self.create_timer(0.05, self.tick)
            self.get_logger().info(VERSION + ': Cross=walk; triangle=stand then Student; square=request roll-to-stand')
            self.get_logger().info('Circle=continuous rolling; neutral vx=0.60, range=0.45..0.75 m/s; yaw=0 or +/-0.02..0.07 rad/s')
            self.get_logger().info(
                f'Gamepad indices: Cross={self.seq.indices[0]}, Triangle={self.seq.indices[1]}, '
                f'Square={self.seq.indices[2]}, R3/estop={self.seq.indices[3]}, '
                f'Options/reset={self.seq.indices[4]}; expected buttons={self.seq.expected_button_count}')

        def latch_stop(self):
            if self.seq.state != 'estop':
                self.generation += 1
                self.seq.state = 'estop'
                self.get_logger().warn('Sequence stopped. Release R3, press Options to clear, then Cross to walk.')

        def stop(self):
            self.latch_stop()
            self.estop.publish(Empty())

        def joy(self, msg):
            self.joy_time = time.monotonic()
            try:
                self.rolling_axes = rolling_command(msg.axes, self.p['rolling_forward_axis'],
                    self.p['rolling_yaw_axis'], self.p['rolling_joystick_deadzone'])
            except ValueError:
                self.rolling_axes = None
                if self.seq.state in ('roll_requested', 'continuous_rolling'):
                    self.stop()
            count = len(msg.buttons)
            if not self.seq.accepts_button_count(count):
                if self.bad_button_count != count:
                    self.get_logger().error(
                        f'Gamepad mapping mismatch: received {count} buttons, expected '
                        f'{self.seq.expected_button_count}. Sequence input blocked; '
                        'verify the receiver mode and emergency-stop mapping before use.')
                    self.bad_button_count = count
            else:
                self.bad_button_count = None
            cross = self.seq.indices[0]
            cross_edge = (self.seq.previous is not None and len(msg.buttons)>cross
                          and msg.buttons[cross] and not self.seq.previous[0])
            previous_state = self.seq.state
            action = self.seq.buttons(msg.buttons)
            if cross_edge:
                self.get_logger().info(f'Cross received: state={previous_state}, action={action}')
            if action == 'stop': self.stop()
            elif action == 'reset':
                if (time.monotonic()-self.controllers_time < .5 and
                        self.pending is None and self.controllers and not any(
                        self.controllers.get(n) == 'active' for n in (self.walk,self.roll))):
                    self.seq.state = 'idle'
            elif action in ('walk','roll'):
                self.switch(action)
            elif action == 'return_walk':
                self.seq.request_walk_return(time.monotonic())
                self.get_logger().info('Cross: waiting for upright, low joint/gyro velocity before returning to Walking')
            elif action == 'stand' and self.controllers.get(self.roll) == 'active':
                if self.stand.get_subscription_count():
                    self.stand.publish(Empty())
                    self.seq.state = 'stand_requested'
                    self.get_logger().info('Square: roll-to-stand requested; existing pitch gate controls handoff')
            elif action == 'continuous_roll' and self.controllers.get(self.roll) == 'active':
                if (self.rolling_axes is not None and self.continuous.get_subscription_count()
                        and time.monotonic()-self.rolling_stage_time < 0.5
                        and self.rolling_stage in ('startup', 'rejected')):
                    self.publish_rolling_velocity()
                    self.continuous.publish(Empty())
                    self.seq.state = 'roll_requested'
                    self.rolling_pending_seen = False
                    self.get_logger().info('Circle: waiting for mature rolling and a continuous target handoff')
                else:
                    self.get_logger().warn('Circle ignored: continuous policy/command/status is not ready')

        def policy_state(self, msg):
            self.rolling_stage, self.rolling_stage_time = msg.data, time.monotonic()
            if self.seq.state == 'roll_requested':
                if msg.data == 'pending':
                    self.rolling_pending_seen = True
                if msg.data == 'continuous':
                    self.seq.state = 'continuous_rolling'
                    self.get_logger().info('PPO continuous rolling takeover confirmed')
                elif msg.data == 'rejected' and self.rolling_pending_seen:
                    self.seq.state = 'rolling'
                    self.get_logger().warn('Takeover timed out or target discontinuity remained too large; startup policy continues. Release and press Circle to retry.')

        def publish_rolling_velocity(self):
            if self.rolling_axes is None:
                return
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = self.rolling_axes
            self.rolling_velocity.publish(msg)

        def joint(self, msg):
            self.joints, self.joint_time = msg, time.monotonic()

        def inertial(self, msg):
            self.imu, self.imu_time = msg, time.monotonic()

        def output(self, msg):
            if self.seq.state == 'starting' and len(msg.data) == 12 and all(map(math.isfinite,msg.data)):
                self.seq.state = 'rolling'

        def stable(self, now, check_startup_pose=True):
            def blocked(reason):
                self.blocked_reason = reason
                return False
            if now-self.joint_time > .25 or now-self.imu_time > .25:
                return blocked(f'stale sensors: joint_age={now-self.joint_time:.2f}s imu_age={now-self.imu_time:.2f}s (limit 0.25s)')
            positions = dict(zip(self.joints.name,self.joints.position))
            velocities = dict(zip(self.joints.name,self.joints.velocity))
            for name, target in zip(self.p['joint_names'],self.p['startup_joint_pos']):
                q, v = positions.get(name,math.nan), velocities.get(name,math.nan)
                if not math.isfinite(q) or not math.isfinite(v): return blocked(f'missing/non-finite joint state: {name}')
                if check_startup_pose and abs(q-target)>self.p['joint_tolerance']:
                    return blocked(f'{name} position error={abs(q-target):.3f}rad exceeds {self.p["joint_tolerance"]}')
                if abs(v)>self.p['velocity_tolerance']:
                    return blocked(f'{name} speed={abs(v):.3f}rad/s exceeds {self.p["velocity_tolerance"]}')
            rates = self.imu.angular_velocity
            q = self.imu.orientation
            values = (q.x,q.y,q.z,q.w)
            norm = sum(v*v for v in values)
            if not all(map(math.isfinite,values)) or norm<1e-8: return blocked('invalid IMU quaternion')
            upright = 1.-2.*(q.x*q.x+q.y*q.y)/norm
            if upright<math.cos(self.p['stand_tilt_tolerance']):
                return blocked(f'body tilt={math.degrees(math.acos(max(-1.,min(1.,upright)))):.1f}deg exceeds {math.degrees(self.p["stand_tilt_tolerance"]):.1f}deg')
            for axis,v in zip('xyz',(rates.x,rates.y,rates.z)):
                if not math.isfinite(v) or abs(v)>self.p['gyro_tolerance']:
                    return blocked(f'gyro {axis}={v:.3f}rad/s exceeds {self.p["gyro_tolerance"]}')
            self.blocked_reason = 'sensors stable; waiting for continuous hold interval'
            return True

        def switch(self, target):
            if self.pending is not None:
                self.blocked_reason = 'controller switch request still pending'
                return
            if not self.switch_client.service_is_ready():
                self.blocked_reason = 'switch_controller service unavailable'
                return
            if time.monotonic()-self.controllers_time > .5:
                self.blocked_reason = 'controller state snapshot stale'
                return
            name = self.walk if target == 'walk' else self.roll
            if target != 'stop' and self.controllers.get(name) != 'inactive':
                self.blocked_reason = f'{name} must be inactive, got {self.controllers.get(name)}'
                return
            if target == 'roll' and self.controllers.get(self.walk) != 'active': return
            if target == 'walk' and self.seq.state == 'waiting_walk':
                if self.controllers.get(self.roll) != 'active' or not self.stable(time.monotonic(),False): return
            request = SwitchController.Request()
            request.activate_controllers = [] if target == 'stop' else [name]
            request.deactivate_controllers = [n for n in (self.walk,self.roll)
                if self.controllers.get(n) == 'active' and (target=='stop' or n!=name)]
            if target=='stop' and not request.deactivate_controllers: return
            request.strictness = SwitchController.Request.STRICT
            request.timeout.sec = 2
            self.get_logger().info(f'Sending switch request: activate={request.activate_controllers}, deactivate={request.deactivate_controllers}')
            self.pending = (self.switch_client.call_async(request),target,self.generation,time.monotonic())
            if target != 'stop': self.seq.state = 'switching'

        def tick(self):
            now = time.monotonic()
            # The private topic never reaches the walking or startup policy.
            # Continue feeding the PPO while Square waits for its stop pitch gate.
            if (self.seq.state in ('rolling', 'roll_requested', 'continuous_rolling',
                                   'stand_requested', 'waiting_walk', 'walk_return_blocked')
                    and now-self.joy_time <= self.p['joy_timeout_seconds']
                    and self.bad_button_count is None):
                self.publish_rolling_velocity()
            if self.list_future is not None and self.list_future.done():
                try:
                    self.controllers = {c.name:c.state for c in self.list_future.result().controller}
                    self.controllers_time = now
                except Exception:
                    self.controllers = {}
                self.list_future = None
            if self.list_future is None and self.list_client.service_is_ready():
                self.list_future = self.list_client.call_async(ListControllers.Request())
            if self.seq.state not in ('idle','estop') and now-self.joy_time>self.p['joy_timeout_seconds']:
                self.stop()
            if self.pending is not None:
                future,target,generation,started = self.pending
                if future.done():
                    self.pending = None
                    # A pre-switch snapshot must never authorize reset or a
                    # second switch after a late controller-manager response.
                    self.controllers_time = -math.inf
                    self.list_future = None
                    try: ok = future.result().ok
                    except Exception: ok = False
                    if generation == self.generation and self.seq.state != 'estop':
                        if ok:
                            self.seq.switched(target,now)
                            self.get_logger().info(f'Controller switch confirmed: {target}')
                        else:
                            self.get_logger().error(f'Controller manager rejected/failed switch: {target}')
                            self.stop()
                    elif self.seq.state == 'estop':
                        self.estop.publish(Empty())
                elif now-started>3.0:
                    self.stop()  # Keep request tracked: a late reply must never enable a policy.
            if self.seq.state == 'estop':
                self.switch('stop')
            elif self.seq.state == 'settling':
                action = self.seq.settle(now, self.controllers.get(self.roll)=='active' and self.stable(now))
                if action=='stop': self.stop()
                elif action=='enable' and self.enable.get_subscription_count():
                    self.enable.publish(Empty())
                    self.seq.state, self.enable_time = 'starting', now
            elif self.seq.state == 'starting' and now-self.enable_time>2.0:
                self.stop()
            elif self.seq.state == 'waiting_walk':
                self.blocked_reason = 'rolling controller not active or controller state snapshot stale'
                stable = (now-self.controllers_time < .5 and
                          self.controllers.get(self.roll)=='active' and self.stable(now,False))
                action = self.seq.walk_return(now,stable)
                if action == 'walk':
                    self.switch('walk')
                elif action == 'cancel':
                    self.seq.state = 'walk_return_blocked'
                    self.get_logger().warn('Walking return blocked: '+self.blocked_reason+'. Current policy continues; Cross retries.')
            self.status.publish(String(data=self.seq.state))
            self.detail.publish(String(data=f'{VERSION}; state={self.seq.state}; {self.blocked_reason}'))

    rclpy.init()
    node = Coordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
