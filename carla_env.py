# carla_env.py
import carla
import random
import numpy as np
import time
import math
from collections import deque

class CarlaEnv:
    def __init__(self, stack_size=4, max_speed_kmh=35, max_episode_steps=300):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.vehicle          = None
        self.collision_sensor = None
        self.camera_sensor    = None
        self.seg_sensor       = None

        self.stack_size  = stack_size
        self.frame_stack = deque(maxlen=stack_size)
        self.seg_stack   = deque(maxlen=stack_size)

        self.max_speed_kmh      = max_speed_kmh
        self.max_episode_steps  = max_episode_steps
        self.current_step       = 0
        self.collision_detected = False

        self._rgb_width  = 84
        self._rgb_height = 84

        # Segmentation update frequency — every 3 ticks to reduce overhead
        self._seg_update_freq = 3
        self._seg_step_count  = 0
        self._last_seg_frame  = np.zeros(
            (self._rgb_height, self._rgb_width, 1), dtype=np.float32
        )

        # Waypoint tracking
        self._current_waypoint  = None
        self._next_waypoint     = None
        self._waypoint_distance = 5.0

    # =========================================================
    # RESET
    # =========================================================
    def reset(self):
        for actor in [self.vehicle, self.collision_sensor,
                      self.camera_sensor, self.seg_sensor]:
            if actor:
                actor.destroy()

        for _ in range(5):
            self.world.tick()

        # Spawn vehicle
        spawn_points = self.world.get_map().get_spawn_points()
        max_attempts = 10
        for attempt in range(max_attempts):
            spawn_point = random.choice(spawn_points)
            vehicle_bp  = self.blueprint_library.filter("model3")[0]
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                break
            except RuntimeError:
                if attempt == max_attempts - 1:
                    raise RuntimeError("Failed to spawn vehicle after multiple attempts")

        # Collision sensor
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision())

        # RGB camera
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self._rgb_width))
        camera_bp.set_attribute("image_size_y", str(self._rgb_height))
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera_sensor.listen(lambda image: self._process_camera(image))

        # Semantic segmentation camera
        seg_bp = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", str(self._rgb_width))
        seg_bp.set_attribute("image_size_y", str(self._rgb_height))
        seg_bp.set_attribute("fov", "110")
        self.seg_sensor = self.world.spawn_actor(
            seg_bp, camera_transform, attach_to=self.vehicle
        )
        self.seg_sensor.listen(lambda image: self._process_segmentation(image))

        # Reset state
        self.frame_stack.clear()
        self.seg_stack.clear()
        self.current_step       = 0
        self.collision_detected = False
        self._seg_step_count    = 0
        self._last_seg_frame    = np.zeros(
            (self._rgb_height, self._rgb_width, 1), dtype=np.float32
        )

        # Initialize waypoints
        self._current_waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        self._next_waypoint = self._current_waypoint.next(
            self._waypoint_distance
        )[0]

        # Tick to get initial sensor data
        for _ in range(10):
            self.world.tick()

        # Fill stacks with zeros
        for _ in range(self.stack_size):
            self.frame_stack.append(
                np.zeros((self._rgb_height, self._rgb_width, 3), dtype=np.float32)
            )
            self.seg_stack.append(
                np.zeros((self._rgb_height, self._rgb_width, 1), dtype=np.float32)
            )

        return self._get_state()

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):
        self.current_step += 1

        throttle = 0.4
        steer    = 0.0
        brake    = 0.0

        if action == 0:
            steer = -0.25
        elif action == 1:
            steer = 0.25
        elif action == 2:
            throttle = 0.0
        elif action == 3:
            throttle = 0.6
        elif action == 4:
            brake = 0.5

        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )
        self.world.tick()

        state = self._get_state()
        reward, done, info = self._compute_reward(action, steer)

        if self.current_step >= self.max_episode_steps:
            done = True

        return state, reward, done, info

    # =========================================================
    # STATE
    # =========================================================
    def _get_state(self):
        # RGB: (stack_size, H, W, 3)
        rgb_stack = np.array(self.frame_stack, dtype=np.float32) / 255.0

        # Segmentation: (stack_size, H, W, 1)
        seg_stack = np.array(self.seg_stack, dtype=np.float32)

        # Combined visual: (stack_size, H, W, 4)
        visual = np.concatenate([rgb_stack, seg_stack], axis=3)

        # Vector: normalized speed and steering
        velocity = self.vehicle.get_velocity()
        speed    = 3.6 * math.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )
        control = self.vehicle.get_control()
        steer   = control.steer

        vector = np.array([
            speed / self.max_speed_kmh,
            steer
        ], dtype=np.float32)

        return {"visual": visual, "vector": vector}

    # =========================================================
    # CAMERA PROCESSING
    # =========================================================
    def _process_camera(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self._rgb_height, self._rgb_width, 4))
        rgb   = array[:, :, :3].astype(np.float32)
        self.frame_stack.append(rgb)

    # =========================================================
    # SEGMENTATION PROCESSING — throttled to every 3 ticks
    # =========================================================
    def _process_segmentation(self, image):
        self._seg_step_count += 1
        if self._seg_step_count % self._seg_update_freq == 0:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((self._rgb_height, self._rgb_width, 4))
            seg   = array[:, :, 2:3].astype(np.float32) / 22.0
            self._last_seg_frame = seg
        # Always append last known seg frame to maintain stack consistency
        self.seg_stack.append(self._last_seg_frame)

    # =========================================================
    # REWARD FUNCTION
    # =========================================================
    def _compute_reward(self, action, steer):
        velocity = self.vehicle.get_velocity()
        speed    = 3.6 * math.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )

        reward         = 0
        done           = False
        dist_to_center = 0.0

        # 1. Smooth speed reward
        target_speed = 20.0
        speed_diff   = abs(speed - target_speed)
        reward      += max(0, 2.0 - 0.1 * speed_diff)

        # 2. Standing still penalty
        if speed < 1.0:
            reward -= 2.0
        elif speed < 5.0:
            reward -= 0.8

        # 3. Lane keeping penalty
        vehicle_location = self.vehicle.get_location()
        waypoint = self.world.get_map().get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is not None:
            lane_center    = waypoint.transform.location
            dist_to_center = math.sqrt(
                (vehicle_location.x - lane_center.x)**2 +
                (vehicle_location.y - lane_center.y)**2
            )
            reward -= min(3.0, dist_to_center * 1.5)
        else:
            dist_to_center = 4.0
            reward        -= 3.0

        # 4. Waypoint progress reward
        if self._next_waypoint is not None:
            dist_to_next = vehicle_location.distance(
                self._next_waypoint.transform.location
            )
            if dist_to_next < self._waypoint_distance:
                reward += 2.0
                next_wps = self._next_waypoint.next(self._waypoint_distance)
                if next_wps:
                    self._next_waypoint = next_wps[0]

        # 5. Heading alignment reward
        if waypoint is not None:
            vehicle_yaw  = math.radians(
                self.vehicle.get_transform().rotation.yaw
            )
            waypoint_yaw = math.radians(
                waypoint.transform.rotation.yaw
            )
            heading_alignment = math.cos(vehicle_yaw - waypoint_yaw)
            reward += heading_alignment * 1.0

        # 6. Survival bonus
        reward += 0.1

        # 7. Collision penalty
        if self.collision_detected:
            reward -= 50
            done    = True

        info = {
            "speed":    speed,
            "lane_dev": dist_to_center,
        }

        return reward, done, info

    # =========================================================
    # COLLISION HANDLER
    # =========================================================
    def _on_collision(self):
        self.collision_detected = True

    # =========================================================
    # CLEANUP
    # =========================================================
    def close(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        for actor in [self.vehicle, self.collision_sensor,
                      self.camera_sensor, self.seg_sensor]:
            if actor:
                actor.destroy()

        self.frame_stack.clear()
        self.seg_stack.clear()