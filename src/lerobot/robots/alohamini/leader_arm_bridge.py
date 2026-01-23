import argparse
import json
import logging
import time
import threading

import websocket  # This is the `websocket-client` library
from lerobot.teleoperators.uni_so100_leader import UniSO100Leader, UniSO100LeaderConfig

# --- Globals to share between threads ---
leader_arm: UniSO100Leader | None = None
ws_app: websocket.WebSocketApp | None = None


def _arm_reader_thread():
    """
    A long-running thread that continuously reads from the arm 
    and sends data over the websocket.
    """
    logging.info("Arm reader thread started.")
    try:
        while True:
            # Get action from the leader arm, which returns keys like "left_shoulder_pan.pos"
            raw_action = leader_arm.get_action()
            
            # Add the "arm_" prefix that the LeKiwiUni robot expects
            action_data = {f"arm_{k}": v for k, v in raw_action.items()}
            
            message = {
                "type": "action",
                "data": action_data
            }
            
            if ws_app:
                ws_app.send(json.dumps(message))
            
            # Control the update frequency
            time.sleep(1 / 50)  # 50Hz
    except Exception as e:
        logging.error(f"Error in arm reader thread: {e}")
        if ws_app:
            ws_app.close()


def on_open(ws):
    logging.info("Connected to robot server.")
    # Start the arm reader thread now that the connection is open
    threading.Thread(target=_arm_reader_thread, daemon=True).start()


def on_message(ws, message):
    # The bridge client does not expect to receive messages, so we do nothing.
    pass


def on_error(ws, error):
    logging.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    logging.info("### WebSocket connection closed ###")


def main():
    parser = argparse.ArgumentParser(description="LeRobot Leader Arm Bridge (Synchronous Client)")
    parser.add_argument("--robot-ip", type=str, required=True, help="IP address of the robot running the web host")
    parser.add_argument("--leader-port", type=str, required=True, help="Serial port of the leader arm")
    parser.add_argument("--leader-id", type=str, default="so100_leader_uni", help="ID for the leader arm")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    global leader_arm, ws_app

    # --- Connect to Leader Arm ---
    leader_arm_config = UniSO100LeaderConfig(port=args.leader_port, id=args.leader_id)
    leader_arm = UniSO100Leader(leader_arm_config)
    
    logging.info(f"Attempting to connect to leader arm on port '{args.leader_port}'...")
    leader_arm.connect()
    if not leader_arm.is_connected:
        logging.error("Failed to connect to the leader arm. Exiting.")
        return
    logging.info("Leader arm connected.")

    # --- Setup and run WebSocket App ---
    ws_url = f"ws://{args.robot_ip}:8000/ws?quality=silent"
    logging.info(f"Connecting to robot at {ws_url}...")
    
    ws_app = websocket.WebSocketApp(ws_url,
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)

    try:
        ws_app.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down bridge.")
    finally:
        if leader_arm.is_connected:
            leader_arm.disconnect()
        logging.info("Bridge disconnected.")


if __name__ == "__main__":
    main()
