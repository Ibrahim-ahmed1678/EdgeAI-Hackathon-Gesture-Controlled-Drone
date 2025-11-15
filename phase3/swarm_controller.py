from pymavlink import mavutil
import time
import math

class DroneSwarm:
    def __init__(self, num_drones=3, base_port=14550):
        self.drones = []
        self.num_drones = num_drones
        
        # Connect to multiple simulated drones
        for i in range(num_drones):
            port = base_port + i
            connection = mavutil.mavlink_connection(f'udp:127.0.0.1:{port}')
            connection.wait_heartbeat()
            self.drones.append({
                'id': i,
                'connection': connection,
                'position': None
            })
            print(f"Drone {i} connected on port {port}")
    
    def arm_all(self):
        """Arm all drones"""
        for drone in self.drones:
            drone['connection'].arducopter_arm()
            print(f"Drone {drone['id']} armed")
        time.sleep(2)
    
    def takeoff_all(self, altitude=10):
        """Takeoff all drones to specified altitude"""
        for drone in self.drones:
            drone['connection'].mav.command_long_send(
                drone['connection'].target_system,
                drone['connection'].target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, altitude
            )
            print(f"Drone {drone['id']} taking off to {altitude}m")
        time.sleep(5)
    
    def formation_grid(self, spacing=5):
        """Move drones into grid formation"""
        positions = [
            (0, 0, 10),           # Center
            (spacing, 0, 10),     # Right
            (-spacing, 0, 10)     # Left
        ]
        
        for i, drone in enumerate(self.drones):
            if i < len(positions):
                x, y, z = positions[i]
                self.goto_position(drone, x, y, z)
    
    def formation_circle(self, radius=10, center_x=0, center_y=0, altitude=10):
        """Move drones into circular formation"""
        for i, drone in enumerate(self.drones):
            angle = (2 * math.pi * i) / self.num_drones
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.goto_position(drone, x, y, altitude)
    
    def orbit_target(self, target_x, target_y, radius=8, altitude=10):
        """Orbit drones around a target position"""
        for i, drone in enumerate(self.drones):
            angle = (2 * math.pi * i) / self.num_drones
            x = target_x + radius * math.cos(angle)
            y = target_y + radius * math.sin(angle)
            self.goto_position(drone, x, y, altitude)
    
    def goto_position(self, drone, x, y, z):
        """Send drone to specific position (NED coordinates)"""
        drone['connection'].mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            drone['connection'].target_system,
            drone['connection'].target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,  # Position only
            x, y, -z,  # NED coordinates (z is negative for altitude)
            0, 0, 0,   # velocity
            0, 0, 0,   # acceleration
            0, 0       # yaw, yaw_rate
        )
    
    def land_all(self):
        """Land all drones"""
        for drone in self.drones:
            drone['connection'].mav.command_long_send(
                drone['connection'].target_system,
                drone['connection'].target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            print(f"Drone {drone['id']} landing")

# Test the swarm
if __name__ == "__main__":
    swarm = DroneSwarm(num_drones=3)
    
    print("Arming drones...")
    swarm.arm_all()
    
    print("Taking off...")
    swarm.takeoff_all(altitude=10)
    
    print("Grid formation...")
    swarm.formation_grid(spacing=5)
    time.sleep(10)
    
    print("Circle formation...")
    swarm.formation_circle(radius=8)
    time.sleep(10)
    
    print("Landing...")
    swarm.land_all()