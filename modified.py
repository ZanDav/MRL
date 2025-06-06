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
from pathlib import Path
import glob
from tqdm import tqdm # <--- [NUOVO] Importa la libreria per la barra di avanzamento

from agent import ENV_KWARGS
from inverse_dynamics_model import IDMAgent

# --- START OF MODIFICATIONS: Hardcoded paths and directories ---
DEFAULT_WEIGHTS_PATH = "VPT-models/IDM/4x_idm.weights"
DEFAULT_MODEL_PATH = "VPT-models/IDM/4x_idm.model"
VIDEO_INPUT_DIRECTORY = "Environments/IDM"
JSON_OUTPUT_DIRECTORY = "/content/Environments/IDM"
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
    "ESC": 0, "back": 0, "drop": 0, "forward": 0, "hotbar.1": 0, "hotbar.2": 0,
    "hotbar.3": 0, "hotbar.4": 0, "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0,
    "hotbar.8": 0, "hotbar.9": 0, "inventory": 0, "jump": 0, "left": 0,
    "right": 0, "sneak": 0, "sprint": 0, "swapHands": 0, "camera": np.array([0, 0]),
    "attack": 0, "use": 0, "pickItem": 0,
}

MESSAGE = """
This script will take videos from {}, predict actions for their frames,
and save predictions to JSONL files in {}.
If a JSONL file with the same name as the video exists in the input directory,
it will be used to compare predictions with true actions.
""".format(VIDEO_INPUT_DIRECTORY, JSON_OUTPUT_DIRECTORY)

CAMERA_SCALER = 360.0 / 2400.0

def json_action_to_env_action(json_action):
    env_action = NOOP_ACTION.copy()
    env_action["camera"] = np.array([0, 0])
    is_null_action = True
    if "keyboard" in json_action and "keys" in json_action["keyboard"]:
        for key in json_action["keyboard"]["keys"]:
            if key in KEYBOARD_BUTTON_MAPPING:
                env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
                is_null_action = False
    if "mouse" in json_action:
        mouse = json_action["mouse"]
        camera_action = env_action["camera"]
        camera_action[0] = mouse.get("dy", 0.0) * CAMERA_SCALER
        camera_action[1] = mouse.get("dx", 0.0) * CAMERA_SCALER
        if mouse.get("dx", 0.0) != 0 or mouse.get("dy", 0.0) != 0: is_null_action = False
        else:
            if abs(camera_action[0]) > 180: camera_action[0] = 0
            if abs(camera_action[1]) > 180: camera_action[1] = 0
        mouse_buttons = mouse.get("buttons", [])
        if 0 in mouse_buttons: env_action["attack"] = 1; is_null_action = False
        if 1 in mouse_buttons: env_action["use"] = 1; is_null_action = False
        if 2 in mouse_buttons: env_action["pickItem"] = 1; is_null_action = False
    return env_action, is_null_action

def predicted_action_to_json_action(predicted_agent_frame_action, tick, prev_mouse_buttons, prev_keyboard_keys):
    json_output = {"mouse": {"x": 640.0, "y": 360.0, "dx": 0.0, "dy": 0.0, "scaledX": 0.0, "scaledY": 0.0, "dwheel": 0.0, "buttons": [], "newButtons": []},
                   "keyboard": {"keys": [], "newKeys": [], "chars": ""}, "hotbar": 0, "tick": tick, "isGuiOpen": False}
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
    if selected_hotbar_slot != -1: json_output["hotbar"] = selected_hotbar_slot
    return json_output

def main(n_batches=10, n_frames=128):
    print(MESSAGE)
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
    required_resolution = ENV_KWARGS["resolution"]
    video_files = glob.glob(f"{VIDEO_INPUT_DIRECTORY}/*.mp4")
    if not video_files:
        print(f"No .mp4 files found in {VIDEO_INPUT_DIRECTORY}")
        return
    Path(JSON_OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # <--- [MODIFICA] Applica tqdm al ciclo principale dei video
    for video_path_str in tqdm(video_files, desc="Processing All Videos"):
        video_path = Path(video_path_str)
        # print(f"\n\nProcessing video: {video_path.name}") # Commentato per un output più pulito

        json_comparison_path_obj = video_path.with_suffix(".jsonl")
        json_comparison_path = str(json_comparison_path_obj) if json_comparison_path_obj.exists() else None
        # if json_comparison_path: print(f"Found comparison JSONL: {json_comparison_path_obj.name}")
        # else: print(f"No comparison JSONL found for {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path.name}")
            continue

        output_jsonl_path = Path(JSON_OUTPUT_DIRECTORY) / f"{video_path.stem}.jsonl"
        all_json_predictions = []
        global_tick = 0
        previous_json_mouse_buttons = []
        previous_json_keyboard_keys = []
        json_data, json_index = None, 0
        correct_counts = {action: 0 for action in NOOP_ACTION.keys()}
        total_frames_for_accuracy, threshold = 0, 1.0

        if json_comparison_path is not None:
            try:
                with open(json_comparison_path) as json_file:
                    json_lines = json_file.readlines()
                    valid_json_lines = [line.strip() for line in json_lines if line.strip() and line.strip() != ","]
                    json_data = json.loads("[" + ",".join(valid_json_lines) + "]")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                # print(f"Warning: Could not load comparison JSONL {json_comparison_path}: {e}")
                json_data = None

        # <--- [MODIFICA] Applica una seconda tqdm al ciclo dei batch per ogni video
        # 'leave=False' fa sì che la barra del singolo video scompaia al termine, mantenendo l'output pulito.
        batch_iterator = tqdm(range(n_batches), desc=f"Video: {video_path.name[:25]}...", leave=False)
        for batch_num in batch_iterator:
            th.cuda.empty_cache()
            frames, recorded_actions_for_batch = [], [] if json_data is not None else None
            frames_to_read_in_batch = n_frames
            if json_data is not None and json_index + n_frames > len(json_data):
                frames_to_read_in_batch = len(json_data) - json_index

            for _ in range(frames_to_read_in_batch):
                ret, frame = cap.read()
                if not ret: break
                if frame.shape[0:2] != tuple(required_resolution)[::-1]:
                    frame = cv2.resize(frame, tuple(required_resolution))
                frames.append(frame[..., ::-1])
                if json_data is not None:
                    if json_index < len(json_data):
                        env_action, _ = json_action_to_env_action(json_data[json_index])
                        recorded_actions_for_batch.append(env_action)
                        json_index += 1
                    else: break
            if not frames: break

            frames_np, n_frames_in_batch = np.stack(frames), len(frames)
            if json_data is not None: total_frames_for_accuracy += n_frames_in_batch
            predicted_actions = agent.predict_actions(frames_np)

            for i in range(n_frames_in_batch):
                current_frame_agent_action = {key: val[0, i] for key, val in predicted_actions.items()}
                json_action_output = predicted_action_to_json_action(
                    current_frame_agent_action, global_tick, previous_json_mouse_buttons, previous_json_keyboard_keys
                )
                all_json_predictions.append(json_action_output)
                previous_json_mouse_buttons = json_action_output["mouse"]["buttons"]
                previous_json_keyboard_keys = json_action_output["keyboard"]["keys"]
                global_tick += 1

                if json_data is not None and recorded_actions_for_batch and i < len(recorded_actions_for_batch):
                    recorded_action = recorded_actions_for_batch[i]
                    for action_name_acc in predicted_actions.keys():
                        if action_name_acc == "camera":
                            if np.all(np.abs(current_frame_agent_action["camera"] - recorded_action["camera"]) < threshold):
                                correct_counts["camera"] += 1
                        elif action_name_acc in recorded_action:
                            if (current_frame_agent_action[action_name_acc] > 0.5) == (recorded_action[action_name_acc] == 1):
                                correct_counts[action_name_acc] += 1
        cap.release()
        with open(output_jsonl_path, "w") as f:
            for line_data in all_json_predictions:
                f.write(json.dumps(line_data) + "\n")
        # print(f"Saved {len(all_json_predictions)} predictions to {output_jsonl_path}")

        if json_data is not None and total_frames_for_accuracy > 0:
            print(f"\n--- Accuracy Riepilogo for {video_path.name} ---")
            for action_name_summary, count in correct_counts.items():
                accuracy = (count / total_frames_for_accuracy) * 100
                print(f"  {action_name_summary:<15}: {count:>5} / {total_frames_for_accuracy} ({accuracy:.2f}%)")
            print("-" * 30)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings and save predictions.")
    parser.add_argument("--n-frames", type=int, default=128, help="Number of frames to process at a time per batch.")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches (n-frames) to process.")
    args = parser.parse_args()
    main(args.n_batches, args.n_frames)
