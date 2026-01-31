#!/usr/bin/env python3
"""
Patch script: Adds new commands to nox_daemon.py
- sensors: Full sensor readout (IMU, touch, sound direction, battery, charging)
- body_state: Current servo angles, movement state
- scan: Look around (sweep head) and report what's detected  
- imu: Raw IMU data
- touch: Touch sensor state
- ears: Sound direction
"""

# This file generates the new commands to be inserted into nox_daemon.py

NEW_COMMANDS = '''

def cmd_sensors():
    """Complete sensor readout."""
    result = {"ts": time.time()}
    
    # Battery
    try:
        with dog_lock:
            v = dog.get_battery_voltage()
            result["battery_v"] = round(v, 2)
            # 2S LiPo: 8.4V full, 6.0V empty
            result["battery_pct"] = max(0, min(100, round((v - 6.0) / (8.4 - 6.0) * 100)))
            # Charging detection: voltage > 8.35V with no load usually means charging
            result["charging"] = v > 8.35
    except Exception as e:
        result["battery_error"] = str(e)
    
    # IMU
    try:
        with dog_lock:
            result["imu"] = {
                "acc": list(dog.accData[:3]) if hasattr(dog, 'accData') else None,
                "gyro": list(dog.accData[3:6]) if hasattr(dog, 'accData') and len(dog.accData) >= 6 else None,
                "pitch": round(dog.pitch, 1) if hasattr(dog, 'pitch') else None,
                "roll": round(dog.roll, 1) if hasattr(dog, 'roll') else None,
            }
    except Exception as e:
        result["imu_error"] = str(e)
    
    # Touch
    try:
        with dog_lock:
            touch = dog.dual_touch.read()
            result["touch"] = touch  # N=none, L=left, R=right, LS=left-slide, RS=right-slide
    except Exception as e:
        result["touch_error"] = str(e)
    
    # Sound Direction
    try:
        with dog_lock:
            detected = dog.ears.isdetected()
            result["sound"] = {
                "detected": detected,
                "direction": dog.ears.read() if detected else None,
            }
    except Exception as e:
        result["sound_error"] = str(e)
    
    # Ultrasonic (may not work)
    try:
        dist = dog.read_distance()
        result["distance_cm"] = dist if dist > 0 else None
    except:
        result["distance_cm"] = None
    
    # System
    import shutil
    total, used, free = shutil.disk_usage("/")
    result["system"] = {
        "hostname": os.uname().nodename,
        "uptime_s": int(float(open("/proc/uptime").read().split()[0])),
        "disk_free_gb": round(free / (1024**3), 1),
        "mem_available_mb": round(int(open("/proc/meminfo").readlines()[2].split()[1]) / 1024),
    }
    
    return result


def cmd_body_state():
    """Current body state: servo angles, posture, movement."""
    result = {}
    
    with dog_lock:
        result["leg_angles"] = list(dog.leg_current_angles) if hasattr(dog, 'leg_current_angles') else None
        result["head_angles"] = list(dog.head_current_angles) if hasattr(dog, 'head_current_angles') else None
        result["tail_angles"] = list(dog.tail_current_angles) if hasattr(dog, 'tail_current_angles') else None
        
        # Check if moving
        result["legs_busy"] = not dog.legs_action_buffer == [] if hasattr(dog, 'legs_action_buffer') else None
        result["head_busy"] = not dog.head_action_buffer == [] if hasattr(dog, 'head_action_buffer') else None
        
        # Posture estimation from leg angles
        try:
            la = dog.leg_current_angles
            if la:
                # Rough posture detection
                if all(abs(a) < 50 for a in la):
                    result["posture"] = "lying"
                elif la[4] > 60 and la[6] < -60:  # hind legs tucked
                    result["posture"] = "sitting"
                else:
                    result["posture"] = "standing"
        except:
            result["posture"] = "unknown"
        
        # Battery & charging
        try:
            v = dog.get_battery_voltage()
            result["battery_v"] = round(v, 2)
            result["battery_pct"] = max(0, min(100, round((v - 6.0) / (8.4 - 6.0) * 100)))
            result["charging"] = v > 8.35
        except:
            pass
    
    return result


def cmd_scan():
    """Scan surroundings by sweeping head left-center-right while taking photos."""
    results = []
    positions = [
        (-40, 0, 0, "left"),
        (0, 0, 0, "center"),
        (40, 0, 0, "right"),
    ]
    
    with dog_lock:
        for yaw, roll, pitch, label in positions:
            dog.head_move([[yaw, roll, pitch]], immediately=True, speed=80)
            time.sleep(0.8)
            
    # Take photo at each position (via camera)
    # For now just return the positions we looked at
    return {"ok": True, "positions_scanned": ["left", "center", "right"]}


def cmd_imu():
    """Raw IMU data."""
    with dog_lock:
        return {
            "acc": list(dog.accData[:3]) if hasattr(dog, 'accData') else None,
            "pitch": round(dog.pitch, 1) if hasattr(dog, 'pitch') else None,
            "roll": round(dog.roll, 1) if hasattr(dog, 'roll') else None,
        }


def cmd_touch():
    """Touch sensor state."""
    with dog_lock:
        touch = dog.dual_touch.read()
        return {
            "touch": touch,
            "touched": touch != "N",
            "side": {"N": "none", "L": "left", "R": "right", "LS": "slide-left", "RS": "slide-right"}.get(touch, touch)
        }


def cmd_ears():
    """Sound direction sensor."""
    with dog_lock:
        detected = dog.ears.isdetected()
        return {
            "detected": detected,
            "direction_deg": dog.ears.read() if detected else None,
        }

'''

# New entries for the COMMANDS dict
NEW_COMMAND_ENTRIES = '''
    "sensors": lambda args: cmd_sensors(),
    "body_state": lambda args: cmd_body_state(),
    "scan": lambda args: cmd_scan(),
    "imu": lambda args: cmd_imu(),
    "touch": lambda args: cmd_touch(),
    "ears": lambda args: cmd_ears(),
'''

print("New commands ready to inject into nox_daemon.py")
print(f"Functions: cmd_sensors, cmd_body_state, cmd_scan, cmd_imu, cmd_touch, cmd_ears")
