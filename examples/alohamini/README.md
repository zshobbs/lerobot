# Web-Based Teleoperation for Aloha Mini (LeKiwi Uni)

This document describes the modern, web-based workflow for teleoperating the Aloha Mini robot and recording data. This system replaces the previous workflow that required running separate Python scripts on both the robot and client machines.

## Overview

The new workflow consists of three main components:
1.  **Robot Web Server (`web_host.py`):** A server that runs directly on the robot. It handles the robot's hardware, streams camera feeds (via WebSocket/JPEG), and receives merged control commands.
2.  **Leader Arm Bridge (`leader_arm_bridge.py`):** A simple, synchronous Python script that runs on your local (client) computer. It connects directly to your leader arm and then acts as a **client** to the **Robot Web Server's** WebSocket, sending arm commands.
3.  **Web Interface:** A web page served by the robot that you open in your browser. It displays video, robot status, visual feedback for keyboard presses, and sends keyboard-based control commands (for base and gantry) to the robot.

---

## How to Run

### Step 1: On the Robot Machine

Start the main web server. This will connect to the robot hardware and begin hosting the web interface. For detailed logging (recommended for debugging), use the `--debug` flag.

```bash
python src/lerobot/robots/alohamini/web_host.py --debug
```

### Step 2: On Your Remote/Client Machine

On the computer where you want to teleoperate, you need to perform the following steps.

#### A. Open the Web Interface (for keyboard control and video feedback)

1.  **Find the Robot's IP Address:** If you don't know it, run `ip addr` on the robot's terminal and find its network IP address (e.g., `192.168.1.123`).
2.  **Open Your Browser:** Open a web browser (like Chrome or Firefox).
3.  **Navigate to the Robot:** In the address bar, type `http://<robot-ip-address>:8000`, replacing `<robot-ip-address>` with the actual IP.

You should now see the camera feeds, robot state, and a 'Keyboard Input' display showing your key presses.

#### B. (Optional) Run the Leader Arm Bridge Script (for arm control)

If you wish to control the robot's arm using a leader arm:

1.  **Find Your Leader Arm's Port:** First, find the serial port device path for your leader arm. You can usually find this by running one of the following commands in your terminal:
    ```bash
    # For Linux/macOS
    ls /dev/tty.*
    ```
    Look for a path that resembles `/dev/tty.usbmodem...` or `/dev/ttyACM...`.

2.  **Run the Bridge Script:** Open a **new terminal** on your remote machine and run the `leader_arm_bridge.py` script.

    **Important:** You need to replace `<robot-ip-address>` with the robot's IP (from Step 2A.1), `/path/to/your/leader/arm` with the serial port path you found, and `<your-leader-id>` with the ID of your calibrated leader arm.

    ```bash
    python examples/alohamini/leader_arm_bridge.py --robot-ip <robot-ip-address> --leader-port /path/to/your/leader/arm --leader-id <your-leader-id>
    ```
    This script will connect directly to your leader arm and then connect as a client to the robot's web server, sending arm commands. The "Bridge" status light on the web page should turn green.

### Step 3: Control the Robot

The web page should now be loaded, and relevant status lights should be green. Commands from the keyboard and the bridge script are automatically merged by the robot server.

*   **Keyboard Controls (from Web Interface):**
    -   `w`: Move forward
    -   `s`: Move backward
    -   `a`: Strafe left
    -   `d`: Strafe right
    -   `q`: Rotate counter-clockwise
    -   `e`: Rotate clockwise
    -   `u`: Lift gantry up
    -   `j`: Lift gantry down

*   **Arm Control (from Leader Arm Bridge):**
    -   Move your leader arm to control the robot's arm.

