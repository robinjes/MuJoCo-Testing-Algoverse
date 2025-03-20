import mujoco as mj
import mujoco.viewer
import time

model = mj.MjModel.from_xml_path("scene5inclinedplane.xml")
data = mj.MjData(model)

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():  # Keep the viewer open
        time.sleep(0.05)
        mj.mj_step(model, data)
        
