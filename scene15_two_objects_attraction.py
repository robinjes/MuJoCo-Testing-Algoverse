import mujoco
import mujoco.viewer
import time

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("two_objects_attraction.xml")
data = mujoco.MjData(model)



with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)  # Step simulation forward
        time.sleep(0.05)  # Delay to slow down the simulation (increase this value for slower time)
        viewer.sync()  # Update viewer