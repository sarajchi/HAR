"""
Offline fusion model (video + IMU) prediction using samples in Temporary_Data.

Important:
    - This script does NOT communicate with the Raspberry Pi or the exoskeleton.
    - It only reads samples from Temporary_Data/Sample_* folders and runs the
      trained fusion model to predict actions (grab / down / walk).
    - IMU and video data are assumed to have been collected beforehand by a
      separate script (e.g. 2_GetData.py) that wrote the Sample_* folders.

Behaviour:
    - Scans Temporary_Data for all Sample_* folders.
    - Processes them sequentially from the lowest sample index to the highest.
    - Uses a 2-sample temporal consistency check: two identical consecutive
      predictions are required before an action is considered "stable".
    - Stable actions obey these constraints:
        * No identical consecutive actions (no Grab→Grab, Down→Down, Walk→Walk).
        * Grab → Walk → Down is allowed; Grab → Walk → Grab is forbidden.
        * Grab → Down is allowed.
        * Down → Grab is allowed.
        * Down → Walk is not accepted as a stable transition.
        * Down → Walk → Grab is effectively treated as Down → Grab (Walk ignored).
    - Prints actions and a summary at the end.
"""

import os
import sys
import time
from pathlib import Path

# ----   # Modifiable variables   ----
action_to_idx = {'down': 0, 'grab': 1, 'walk': 2}   # Action to index mapping

PROJECT_ROOT = Path(__file__).resolve().parent
root_directory = PROJECT_ROOT / "Temporary_Data"    # Directory where Sample_* folders are stored

prediction_threshold = 2                            # two consistent predictions required
# ------------------------------------


def print_action_banner(action: str) -> None:
    """Print Grab/Walk/Down in bold and colour to highlight detection."""
    colours = {
        "Grab": "\033[93m",           # Yellow text
        "Down": "\033[95m",           # Magenta text
        "Walk": "\033[44m\033[97m",   # Blue background + white text
    }

    reset = "\033[0m"
    colour = colours.get(action, "")
    text = action.upper()

    if action in {"Grab", "Down"}:
        extra = "  (PAUSE 6 SECONDS)"
    else:
        extra = ""

    print(f"\n{colour}\033[1m=============== {text} DETECTED{extra} ==============={reset}\n")


try:
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
except ModuleNotFoundError as Err:
    missing_module = str(Err).replace('No module named ', '').replace("'", '')
    sys.exit(f'No module named {missing_module} – try: pip install {missing_module}')

try:
    from Imports.InferenceDataloader import HAR_Inference_DataSet
    from Imports.Functions import model_exist, all_the_same
    from Imports.Models.MoViNet.config import _C as config
    from Imports.Models.fusion import FusionModel
except ModuleNotFoundError:
    sys.exit('Missing Imports folder. Make sure you are in the correct project directory.')


def make_prediction(dataset, model, device) -> int:
    """Run one forward pass of the fusion model on the given dataset sample."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    with torch.no_grad():
        for video_frames, imu_data in loader:
            video_frames, imu_data = video_frames.to(device), imu_data.to(device)
            predicted = torch.argmax(model(video_frames, imu_data))
    return predicted.item()


def main() -> None:
    if not model_exist():
        sys.exit("No model to load. Aborting.")

    start_tracking_time = time.time()

    # ----- Transform for video frames -----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ----- Load model -----
    model_dir = PROJECT_ROOT / "Model to Load"
    if not model_dir.exists() or not any(model_dir.iterdir()):
        sys.exit(f"'Model to Load' directory is missing or empty at: {model_dir}")

    model_file = next(model_dir.iterdir())
    model_path = model_dir / model_file.name

    model_name = model_file.name
    if model_name.endswith('.pt'):
        model_name = model_name.replace('.pt', '')
    elif model_name.endswith('.pht'):
        model_name = model_name.replace('.pht', '')

    print(f"Loading {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}\n")

    idx_to_action = {v: k for k, v in action_to_idx.items()}  # Reverse mapping

    model = FusionModel(
        config.MODEL.MoViNetA0,
        num_classes=3,
        lstm_input_size=12,
        lstm_hidden_size=512,
        lstm_num_layers=2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f'\033cProgramme running   ctrl + C to stop\n\nLoading {model_name}\nUsing {device}\n\n\n')

    # -------- OFFLINE: get all sample folders in order --------
    if not root_directory.exists():
        sys.exit(f"Temporary_Data folder not found at: {root_directory}")

    sample_dirs = sorted(
        [d for d in os.listdir(root_directory) if d.startswith("Sample_")],
        key=lambda s: int(s.replace("Sample_", "")),
    )

    if not sample_dirs:
        sys.exit(f"No Sample_* folders found in {root_directory}")

    tracking = [0 for _ in action_to_idx]  # Count for each action index

    # history of *stable* actions, e.g. ["grab", "walk", "down", ...]
    stable_history: list[str] = []

    # rolling window of last N predictions (as action strings, lowercase)
    prediction_save = [''] * prediction_threshold

    first_sample_num = sample_dirs[0]
    last_sample_num = sample_dirs[-1]

    try:
        for sample_name in sample_dirs:
            # Build dataset for THIS sample (HAR_Inference_DataSet must accept target_sample)
            dataset = HAR_Inference_DataSet(
                root_dir=str(root_directory),
                transform=transform,
                target_sample=sample_name,
            )

            try:
                prediction_idx = make_prediction(dataset, model, device)
            except FileNotFoundError:
                print(f'Folder {sample_name} got deleted while reading – skipping.')
                continue

            tracking[prediction_idx] += 1

            # Shift prediction history
            for i in range(prediction_threshold, 1, -1):
                prediction_save[-i] = prediction_save[-i + 1]
            prediction_save[-1] = idx_to_action.get(prediction_idx, 'unknown')

            same_flag = all_the_same(prediction_save)[0]

            if same_flag:
                current_action = prediction_save[-1]  # 'grab', 'down', or 'walk'

                # ---------- transition constraints ----------
                accept = True

                if stable_history:
                    last = stable_history[-1]

                    # 1) no identical consecutive actions
                    if current_action == last:
                        accept = False
                    else:
                        # 2) Down -> Walk is not allowed as a stable transition
                        if last == 'down' and current_action == 'walk':
                            accept = False

                        # 3) Grab/Down alternation when passing through Walk
                        if len(stable_history) >= 2 and last == 'walk':
                            prev = stable_history[-2]
                            if prev == 'grab' and current_action == 'grab':
                                # forbid Grab -> Walk -> Grab
                                accept = False
                            elif prev == 'down' and current_action == 'down':
                                # forbid Down -> Walk -> Down
                                accept = False
                # if stable_history is empty, always accept first stable action

                if accept:
                    stable_history.append(current_action)

                    nice_label = current_action.capitalize()
                    # banner + optional pause
                    if current_action in {"grab", "down"}:
                        print_action_banner(nice_label)
                        time.sleep(6)
                    else:
                        print_action_banner(nice_label)

                    print(
                        f'{sample_name} : {nice_label}  '
                        f'(New stable action at {round(time.time() - start_tracking_time, 2)}s)'
                    )
                else:
                    # Stable but violates transition rules → ignore as action
                    print(prediction_save)
                    print(
                        f'{sample_name} : {current_action} '
                        f'(stable but rejected by transition rules)'
                    )

            else:
                # Not yet stable: just show rolling history and current prediction
                print(prediction_save)
                print(f'{sample_name} : {idx_to_action.get(prediction_idx, "unknown")}')

    except KeyboardInterrupt:
        pass

    # -------- summary --------
    num_of_predictions = sum(tracking)
    num_first = int(first_sample_num.replace('Sample_', ''))
    num_last = int(last_sample_num.replace('Sample_', ''))

    print(f'num_first : {num_first}\nnum_last : {num_last}\nnum of prediction : {num_of_predictions}')
    end_text_prediction = 's' if num_of_predictions != 1 else ''
    print(
        f'\nThere were a total of {num_of_predictions} prediction{end_text_prediction}, '
        f'with {(num_last - num_first + 1) - num_of_predictions} missed'
    )
    for action, idx in action_to_idx.items():
        print(f'{tracking[idx]} raw model outputs for {action}')

    print('\nOffline mode complete: processed all Sample_* folders, no UDP, no RPi communication.')
    print('Stable action rules implemented:')
    print('  • No identical consecutive actions.')
    print('  • Grab → Walk → Down is allowed; Grab → Walk → Grab is rejected.')
    print('  • Grab → Down is allowed.')
    print('  • Down → Grab is allowed.')
    print('  • Down → Walk is not counted as a stable state.')
    print('  • Down → Walk → Grab is treated as Down → Grab (Walk ignored).')


if __name__ == "__main__":
    print("\033cStarting ...\n")  # Clear Terminal
    print("⚠️  This offline fusion script reads Sample_* folders from Temporary_Data.")
    print("    It will process ALL samples sequentially from Sample_1 onwards.\n")
    main()
