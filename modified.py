#@title MODIFIED - Hardcoded paths, processes all mp4s, optional comparison

# NOTE: this is _not_ the original code of IDM!
# As such, while it is close and seems to function well,
# its performance might be bit off from what is reported
# in the paper.

from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
import json
import torch as th
from pathlib import Path # Added for path manipulation
import glob # For finding video files

from agent import ENV_KWARGS
from inverse_dynamics_model import IDMAgent

# --- START OF MODIFICATIONS: Hardcoded paths and directories ---
DEFAULT_WEIGHTS_PATH = "VPT-models/IDM/4x_idm.weights"
DEFAULT_MODEL_PATH = "VPT-models/IDM/4x_idm.model"
VIDEO_INPUT_DIRECTORY = "Environments/IDM"
JSON_OUTPUT_DIRECTORY = "/content/Environments/IDM" # Output directory for predicted JSONLs
# --- END OF MODIFICATIONS ---

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

REVERSE_KEYBOARD_BUTTON_MAPPING = {v: k for k, v in KEYBOARD_BUTTON_MAPPING.items()}

NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

MESSAGE = """
This script will take videos from {}, predict actions for their frames,
and save predictions to JSONL files in {}.
If a JSONL file with the same name as the video exists in the input directory,
it will be used to compare predictions with true actions.

Press 'q' on the OpenCV window to quit processing the current video.
Press any other button to proceed to the next frame of the current video.
""".format(VIDEO_INPUT_DIRECTORY, JSON_OUTPUT_DIRECTORY) # MODIFIED: Updated message

CAMERA_SCALER = 360.0 / 2400.0

def json_action_to_env_action(json_action):
    env_action = NOOP_ACTION.copy()
    env_action["camera"] = np.array([0, 0])
    is_null_action = True
    if "keyboard" in json_action and "keys" in json_action["keyboard"]:
        keyboard_keys = json_action["keyboard"]["keys"]
        for key in keyboard_keys:
            if key in KEYBOARD_BUTTON_MAPPING:
                env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
                is_null_action = False

    if "mouse" in json_action:
        mouse = json_action["mouse"]
        camera_action = env_action["camera"]
        camera_action[0] = mouse.get("dy", 0.0) * CAMERA_SCALER # Use .get for safety
        camera_action[1] = mouse.get("dx", 0.0) * CAMERA_SCALER # Use .get for safety

        if mouse.get("dx", 0.0) != 0 or mouse.get("dy", 0.0) != 0:
            is_null_action = False
        else:
            if abs(camera_action[0]) > 180:
                camera_action[0] = 0
            if abs(camera_action[1]) > 180:
                camera_action[1] = 0

        mouse_buttons = mouse.get("buttons", []) # Use .get for safety
        if 0 in mouse_buttons:
            env_action["attack"] = 1
            is_null_action = False
        if 1 in mouse_buttons:
            env_action["use"] = 1
            is_null_action = False
        if 2 in mouse_buttons:
            env_action["pickItem"] = 1
            is_null_action = False
    return env_action, is_null_action

def predicted_action_to_json_action(predicted_agent_frame_action, tick, prev_mouse_buttons, prev_keyboard_keys):
    json_output = {
        "mouse": {
            "x": 640.0, "y": 360.0,
            "dx": 0.0, "dy": 0.0,
            "scaledX": 0.0, "scaledY": 0.0, "dwheel": 0.0,
            "buttons": [], "newButtons": []
        },
        "keyboard": {
            "keys": [], "newKeys": [], "chars": ""
        },
        "hotbar": 0,
        "tick": tick,
        "isGuiOpen": False
    }
    if CAMERA_SCALER != 0:
        json_output["mouse"]["dy"] = predicted_agent_frame_action["camera"][0] / CAMERA_SCALER
        json_output["mouse"]["dx"] = predicted_agent_frame_action["camera"][1] / CAMERA_SCALER
    else:
        json_output["mouse"]["dy"] = 0.0
        json_output["mouse"]["dx"] = 0.0

    current_mouse_buttons = []
    if predicted_agent_frame_action.get("attack", 0) > 0.5: current_mouse_buttons.append(0)
    if predicted_agent_frame_action.get("use", 0) > 0.5: current_mouse_buttons.append(1)
    if predicted_agent_frame_action.get("pickItem", 0) > 0.5: current_mouse_buttons.append(2)
    json_output["mouse"]["buttons"] = sorted(list(set(current_mouse_buttons)))
    json_output["mouse"]["newButtons"] = sorted(list(set(b for b in current_mouse_buttons if b not in prev_mouse_buttons)))

    current_keyboard_keys = []
    for agent_action_name, pred_val in predicted_agent_frame_action.items():
        if agent_action_name in REVERSE_KEYBOARD_BUTTON_MAPPING and pred_val > 0.5:
            current_keyboard_keys.append(REVERSE_KEYBOARD_BUTTON_MAPPING[agent_action_name])
    json_output["keyboard"]["keys"] = sorted(list(set(current_keyboard_keys)))
    json_output["keyboard"]["newKeys"] = sorted(list(set(k for k in current_keyboard_keys if k not in prev_keyboard_keys)))

    selected_hotbar_slot = -1
    for i in range(1, 10):
        if predicted_agent_frame_action.get(f"hotbar.{i}", 0) > 0.5:
            selected_hotbar_slot = i - 1
            break
    if selected_hotbar_slot != -1:
         json_output["hotbar"] = selected_hotbar_slot
    return json_output

# MODIFIED: Main function now takes n_batches, n_frames, and uses hardcoded paths
def main(n_batches=10, n_frames=128):
    print(MESSAGE)

    # --- START OF MODIFICATIONS: Load agent once ---
    model_path = DEFAULT_MODEL_PATH
    weights_path = DEFAULT_WEIGHTS_PATH

    print(f"Loading model from: {model_path}")
    print(f"Loading weights from: {weights_path}")

    agent_parameters = pickle.load(open(model_path, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights_path)
    # --- END OF MODIFICATIONS ---

    required_resolution = ENV_KWARGS["resolution"]

    # --- START OF MODIFICATIONS: Loop through video files ---
    video_files = glob.glob(f"{VIDEO_INPUT_DIRECTORY}/*.mp4")
    if not video_files:
        print(f"No .mp4 files found in {VIDEO_INPUT_DIRECTORY}")
        return

    Path(JSON_OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    for video_path_str in video_files:
        video_path = Path(video_path_str)
        print(f"\n\nProcessing video: {video_path.name}")

        # Optionally, look for a corresponding JSONL file for comparison
        json_comparison_path_obj = video_path.with_suffix(".jsonl")
        json_comparison_path = str(json_comparison_path_obj) if json_comparison_path_obj.exists() else None
        if json_comparison_path:
            print(f"Found comparison JSONL: {json_comparison_path_obj.name}")
        else:
            print(f"No comparison JSONL found for {video_path.name} (expected {json_comparison_path_obj.name})")


        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path.name}")
            continue

        output_jsonl_path = Path(JSON_OUTPUT_DIRECTORY) / f"{video_path.stem}.jsonl"
        all_json_predictions = []
        global_tick = 0
        previous_json_mouse_buttons = []
        previous_json_keyboard_keys = []

        json_data = None # For comparison data
        json_index = 0
        correct_counts = {action: 0 for action in NOOP_ACTION.keys()}
        total_frames_for_accuracy = 0
        threshold = 1.0 # For camera accuracy

        if json_comparison_path is not None:
            try:
                with open(json_comparison_path) as json_file:
                    json_lines = json_file.readlines()
                    valid_json_lines = [line.strip() for line in json_lines if line.strip() and line.strip() != ","]
                    json_data_str = "[" + ",".join(valid_json_lines) + "]"
                    json_data = json.loads(json_data_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSONL from {json_comparison_path}: {e}")
                json_data = None # Proceed without comparison
            except FileNotFoundError:
                print(f"Comparison file {json_comparison_path} not found.")
                json_data = None # Proceed without comparison


        for batch_num in range(n_batches):
            th.cuda.empty_cache()
            # print(f"--- Video: {video_path.name} - Batch {batch_num + 1}/{n_batches}: Loading frames ---")
            frames = []
            recorded_actions_for_batch = [] if json_data is not None else None

            frames_to_read_in_batch = n_frames
            if json_data is not None and json_index + n_frames > len(json_data):
                frames_to_read_in_batch = len(json_data) - json_index

            for _ in range(frames_to_read_in_batch):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.shape[0] != required_resolution[1] or frame.shape[1] != required_resolution[0]:
                    # print(f"Warning: Frame resolution {frame.shape[1]}x{frame.shape[0]} does not match required {required_resolution[0]}x{required_resolution[1]}. Resizing.")
                    frame = cv2.resize(frame, tuple(required_resolution))
                frames.append(frame[..., ::-1]) # BGR to RGB

                if json_data is not None:
                    if json_index < len(json_data):
                        env_action, _ = json_action_to_env_action(json_data[json_index])
                        recorded_actions_for_batch.append(env_action)
                        json_index += 1
                    else:
                        break
            if not frames:
                # print(f"No more frames to process for {video_path.name} or video ended.")
                break

            frames_np = np.stack(frames)
            n_frames_in_batch = len(frames_np)

            if json_data is not None:
                total_frames_for_accuracy += n_frames_in_batch

            # print(f"--- Video: {video_path.name} - Predicting actions for {n_frames_in_batch} frames ---")
            predicted_actions = agent.predict_actions(frames_np)

            quit_video_processing = False
            for i in range(n_frames_in_batch):
                current_frame_agent_action = {}
                for action_name_key, action_values_for_batch in predicted_actions.items():
                    current_prediction_value = action_values_for_batch[0, i]
                    current_frame_agent_action[action_name_key] = current_prediction_value

                json_action_output = predicted_action_to_json_action(
                    current_frame_agent_action,
                    global_tick,
                    previous_json_mouse_buttons,
                    previous_json_keyboard_keys
                )
                all_json_predictions.append(json_action_output)
                previous_json_mouse_buttons = json_action_output["mouse"]["buttons"]
                previous_json_keyboard_keys = json_action_output["keyboard"]["keys"]
                global_tick += 1

                # --- Visualization and Accuracy (if json_comparison_path provided) ---
                # display_frame = frames_np[i][..., ::-1].copy() # RGB to BGR for cv2
                # text_y_offset = 10

                if json_data is not None and recorded_actions_for_batch and i < len(recorded_actions_for_batch):
                    recorded_action = recorded_actions_for_batch[i]
                    # cv2.putText(display_frame, "Pred (True)", (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    # text_y_offset += 15
                    for action_name_acc in predicted_actions.keys():
                        if action_name_acc == "camera":
                            pred_camera = current_frame_agent_action["camera"]
                            true_camera = recorded_action["camera"]
                            if np.all(np.abs(pred_camera - true_camera) < threshold):
                                correct_counts["camera"] += 1
                        elif action_name_acc in recorded_action:
                            pred_binary = current_frame_agent_action[action_name_acc] > 0.5
                            true_binary = recorded_action[action_name_acc] == 1
                            if pred_binary == true_binary:
                                correct_counts[action_name_acc] += 1

                # Display predicted actions on frame (Commented out for batch processing)
                # for y_disp, (action_name_disp, _) in enumerate(current_frame_agent_action.items()):
                #     pred_val_disp = current_frame_agent_action[action_name_disp]
                #     true_val_disp_str = ""
                #     if json_data is not None and recorded_actions_for_batch and i < len(recorded_actions_for_batch) and action_name_disp in recorded_actions_for_batch[i]:
                #         true_val_disp = recorded_actions_for_batch[i][action_name_disp]
                #         if isinstance(true_val_disp, np.ndarray):
                #              true_val_disp_str = f" ({np.array2string(true_val_disp, formatter={'float_kind':lambda x: '%.1f' % x})})"
                #         else:
                #             true_val_disp_str = f" ({true_val_disp})"

                #     if isinstance(pred_val_disp, np.ndarray):
                #         pred_str = np.array2string(pred_val_disp, formatter={'float_kind':lambda x: "%.1f" % x})
                #         cv2.putText(display_frame, f"{action_name_disp}: {pred_str}{true_val_disp_str}", (10, text_y_offset + y_disp * 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
                #     else:
                #         cv2.putText(display_frame, f"{action_name_disp}: {pred_val_disp:.2f}{true_val_disp_str}", (10, text_y_offset + y_disp * 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

                # cv2.imshow(f"MineRL IDM - {video_path.name}", display_frame) # MODIFIED window title
                # key = cv2.waitKey(0) # Changed to 1 for faster processing, 0 for frame-by-frame
                # if key & 0xFF == ord('q'):
                #     print(f"Quitting processing for {video_path.name} early.")
                #     quit_video_processing = True
                #     break

            # if quit_video_processing:
            #     break

        cap.release()

        with open(output_jsonl_path, "w") as f:
            for line_data in all_json_predictions:
                f.write(json.dumps(line_data) + "\n")
        print(f"Saved {len(all_json_predictions)} predictions for {video_path.name} to {output_jsonl_path}")

        if json_data is not None and total_frames_for_accuracy > 0:
            print(f"\nAccuracy Riepilogo for {video_path.name}:")
            for action_name_summary in NOOP_ACTION.keys():
                if action_name_summary in correct_counts:
                     accuracy = (correct_counts[action_name_summary] / total_frames_for_accuracy) * 100 if total_frames_for_accuracy > 0 else 0
                     print(f"  {action_name_summary}: {correct_counts[action_name_summary]} / {total_frames_for_accuracy} ({accuracy:.2f}%)")
            print(f"  Nota: per 'camera', la previsione è corretta se entro {threshold} dai valori reali.")
            print(f"  Per azioni binarie, la previsione > 0.5 è confrontata con il valore vero (1).")
        # --- END OF MODIFICATIONS for video loop ---
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings and save predictions.")
    # --- START OF MODIFICATIONS: Removed unused arguments ---
    # parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    # parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    # parser.add_argument("--video-path", type=str, required=True, help="Path to a .mp4 file (Minecraft recording).")
    # parser.add_argument("--jsonl-path", type=str, required=False, default=None, help="Optional path to a .jsonl file (true actions for comparison).")
    # --- END OF MODIFICATIONS ---
    parser.add_argument("--n-frames", type=int, default=128, help="Number of frames to process at a time per batch.")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches (n-frames) to process. Set high to process more of the video.")
    args = parser.parse_args()

    # --- MODIFIED: Call main with only relevant args ---
    main(args.n_batches, args.n_frames)