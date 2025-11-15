from pymavlink import mavutil

# Connect to simulator
master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
master.wait_heartbeat()
print("Heartbeat received! Drone is alive.")

# Arm and takeoff test
master.arducopter_arm()
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0, 0, 0, 0, 0, 0, 0, 10  # Takeoff to 10m
)