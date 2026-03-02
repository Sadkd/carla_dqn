import carla
import numpy as np
import cv2
import random
import time
from collections import deque


class CarlaEnv:

    SHOW_CAM = False
    STEER_AMT = 1.0
    im_width = 84      # 96×96 qualite|| 128×128 haute qualite || 160×120 large vision || non carre 160*84
    im_height = 84

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.model_3 = self.blueprint_library.filter("model3")[0]

        self.frame_stack = 4
        self.frames = deque(maxlen=self.frame_stack)

        # Initialize actors
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None

        self.front_camera = None
        self.collision_hist = []

    # --------------------------------------------------
    # DESTROY ALL ACTORS CLEANLY
    # --------------------------------------------------
    def destroy_actors(self):

        actors = [
            self.camera,
            self.collision_sensor,
            self.vehicle
        ]

        for actor in actors:
            if actor is not None:
                try:
                    actor.stop() if hasattr(actor, "stop") else None
                    actor.destroy()
                except:
                    pass

        self.camera = None
        self.collision_sensor = None
        self.vehicle = None

    # --------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------
    def reset(self):

        # Destroy previous actors before spawning new ones
        self.destroy_actors()

        self.collision_hist = []
        self.front_camera = None
        self.frames.clear()

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        self.vehicle = None

        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.try_spawn_actor(self.model_3, spawn_point)
                if self.vehicle is not None:
                    break
            except:
                continue

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle at any spawn point.")


        # ---------------- Camera ----------------
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", f"{self.im_width}")
        camera_bp.set_attribute("image_size_y", f"{self.im_height}")
        camera_bp.set_attribute("fov", "110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(
            camera_bp,
            transform,
            attach_to=self.vehicle
        )

        self.camera.listen(lambda data: self.process_img(data))

        # ---------------- Collision Sensor ----------------
        collision_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_sensor_bp,
            carla.Transform(),
            attach_to=self.vehicle,
        )

        self.collision_sensor.listen(
            lambda event: self.collision_hist.append(event)
        )

        # Wait for first camera frame
        while self.front_camera is None:
            time.sleep(0.01)

        state = self.front_camera

        # Fill frame stack initially
        for _ in range(self.frame_stack):
            self.frames.append(state)

        stacked_state = np.stack(self.frames, axis=0)

        return stacked_state

    # --------------------------------------------------
    # IMAGE PROCESSING
    # --------------------------------------------------
    def process_img(self, image):
        img = np.array(image.raw_data)
        img = img.reshape((self.im_height, self.im_width, 4))
        img = img[:, :, :3]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float32) / 255.0

        self.front_camera = img_gray

    # --------------------------------------------------
    # STEP FUNCTION
    # --------------------------------------------------
    def step(self, action):

        if action == 0:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0)
            )
        elif action == 1:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=-0.5)
            )
        elif action == 2:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0.5)
            )

        time.sleep(0.05)

        collision = len(self.collision_hist) > 0
        done = False

        if collision:
            reward = -100
            done = True
        else:
            velocity = self.vehicle.get_velocity()
            kmh = 3.6 * np.sqrt(
                velocity.x**2 +
                velocity.y**2 +
                velocity.z**2
            )
            reward = kmh / 10

        state = self.front_camera
        self.frames.append(state)
        stacked_state = np.stack(self.frames, axis=0)

        info = {"collision": collision}

        return stacked_state, reward, done, info

    # --------------------------------------------------
    # CLOSE ENVIRONMENT
    # --------------------------------------------------
    def close(self):
        self.destroy_actors()
