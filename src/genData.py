import time
import queue
from datetime import datetime

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


# XML model string
XML = """
<mujoco model="rangefinder array">
    <worldbody>
        <geom name="floor" type="plane" size="2 2 .01"/>
        <geom size=".4"/>
        <body name="1" pos="0 0 1.2">
            <light pos="0.3 0 0" dir="1 0 0"/>
        <joint type="hinge" axis="1 0 0" limited="true" range="-20 20"/>
        <joint type="hinge" axis="0 1 0" limited="true" range="55 110"/>
        <joint type="hinge" axis="0 0 1" limited="true" range="-20 20"/>
        <geom type="box" size=".05 .05 .05" pos=".1 0 0"/>
        <camera name="cam" pos="0.2 0 0" euler="0 -90 0"/>
        <replicate count="4" euler="0 4 0">
            <replicate count="4" euler="0 0 -4">
            <site name="rf" pos=".16 0 0" zaxis="1 0 0"/>
            </replicate>
        </replicate>
        </body>
    </worldbody>
</mujoco>
"""

# Initialize Mujoco model and data
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

def capture_rgb_image(renderer, data, camera_name) -> np.ndarray:
    """Return the captured RGB image."""
    renderer.update_scene(data, camera=camera_name)
    return renderer.render()

def capture_depth_image(renderer, data, camera_name) -> np.ndarray:
    """Return the captured depth image."""
    renderer.update_scene(data, camera=camera_name)
    renderer.enable_depth_rendering()
    depth_image = renderer.render()
    renderer.disable_depth_rendering()
    depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255
    return np.uint8(depth_image)

def save_image(image: np.ndarray, path: str) -> None:
    """Save the image to a file."""
    im = Image.fromarray(image)
    im.save(path)

# Simulation state variables
is_paused = True
key_queue = queue.Queue()

def keyboard_callback(key) -> None:
    """
    Method for the keyboard callback. This method should not be called explicitly.

    Parameters:
    - key: Keyboard input.

    Returns:
    - None.
    """
    global is_paused, renderer, data
    if key == glfw.KEY_SPACE:
        is_paused = not is_paused
        print("Simulation paused." if is_paused else "Simulation resumed.")
    elif key == glfw.KEY_PERIOD:
        print("Shoot! I just took a picture")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        rgb_save_path = f"data/mujoco/{timestamp}_rgb.png"
        save_image(capture_rgb_image(renderer, data, "cam"), rgb_save_path)
        print(f"Saved RGB image to: {rgb_save_path}")

        depth_save_path = f"data/mujoco/{timestamp}_depth.png"
        save_image(capture_depth_image(renderer, data, "cam"), depth_save_path)
        print(f"Saved depth image to: {depth_save_path}")

timestamp = 0
limits = [(np.deg2rad(-20), np.deg2rad(20)), (np.deg2rad(55), np.deg2rad(110)), (np.deg2rad(-20), np.deg2rad(20))]
inti_config = np.array([np.random.uniform(low, high) for low, high in limits])
data.qpos = inti_config
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data, key_callback=lambda key: key_queue.put(key)) as viewer:
    while viewer.is_running():
        step_start = time.time()

        while not key_queue.empty():
            keyboard_callback(key_queue.get())

        inti_config = np.array([np.random.uniform(low, high) for low, high in limits])
        data.qpos = inti_config

        if not is_paused:            
            rgb_save_path = f"data/mujoco/rgb/{timestamp}.png"
            save_image(capture_rgb_image(renderer, data, "cam"), rgb_save_path)

            depth_save_path = f"data/mujoco/depth/{timestamp}.png"
            save_image(capture_depth_image(renderer, data, "cam"), depth_save_path)
            
            mujoco.mj_step(model, data)

            timestamp += 1
        
        if timestamp == 1000:
            break

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
