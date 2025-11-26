
"""
Data acquisition script for IMU-only recording.

- Connects to XIMU3 IMUs over UDP.
- Streams IMU data via callbacks.
- Saves IMU samples into rolling per-sample folders.

Each sample directory has:
    Sample_<n>/
        imu.csv    # sequence_length rows of IMU data

This script is intended as companion code for the associated publication.
"""

if __name__ == "__main__":
    print("\033cStarting ...\n")  # Clear terminal

import csv
import os
import sys
from time import sleep, time

# ----------------------- User-configurable parameters ----------------------- #

ROOT_DIRECTORY: str = "Temporary_Data"   # Directory where sample folders are stored
FPS: int = 30                           # Target IMU sample rate (rows per second)
BUFFER: int = 1500                      # Maximum number of sample folders kept (rolling buffer)
CLEAN_FOLDER: bool = False              # If True, delete all temporary folders on exit
WIFI_TO_CONNECT: str = "Upper_Limb_Exo" # Expected Wi-Fi SSID for Raspberry Pi and IMUs ('' to disable)
WINDOW_SIZE: int = 200                  # Number of IMU lines shown in the terminal window
PRINT_IMU: bool = True                  # If True, print IMU data to terminal

# --------------------------------------------------------------------------- #

try:
    import ximu3
except ModuleNotFoundError as err:
    missing_module = str(err).replace("No module named ", "").replace("'", "")
    sys.exit(f"No module named {missing_module}. Try: pip install {missing_module}")

try:
    from Imports.Functions import format_time, connected_wifi, ask_yn
except ModuleNotFoundError:
    sys.exit("Missing Imports folder. Make sure you are in the correct project root directory.")

LINE_UP = "\033[1A"
LINE_CLEAR = "\x1b[2K"

# Global IMU values updated via callbacks
gyr_x_1 = gyr_y_1 = gyr_z_1 = 0.0
acc_x_1 = acc_y_1 = acc_z_1 = 0.0
gyr_x_2 = gyr_y_2 = gyr_z_2 = 0.0
acc_x_2 = acc_y_2 = acc_z_2 = 0.0


class Connection:
    """
    Wrapper around a single XIMU3 UDP connection.

    Opens the connection, pings the device, and registers an inertial callback.
    """

    def __init__(self, connection_info):
        self._connection = ximu3.Connection(connection_info)

        if self._connection.open() != ximu3.RESULT_OK:
            sys.exit("Unable to open connection " + connection_info.to_string())

        ping_response = self._connection.ping()
        if ping_response.result != ximu3.RESULT_OK:
            print("Ping failed for " + connection_info.to_string())
            raise AssertionError

        self._prefix = ping_response.serial_number
        self._connection.add_inertial_callback(self._inertial_callback)

    def close(self) -> None:
        self._connection.close()

    def send_command(self, key: str, value=None) -> None:
        """
        Send a JSON command to the device.
        """
        if value is None:
            value_str = "null"
        elif isinstance(value, bool):
            value_str = str(value).lower()
        elif isinstance(value, str):
            value_str = f"\"{value}\""
        else:
            value_str = str(value)

        command = f'{{"{key}":{value_str}}}'
        responses = self._connection.send_commands([command], 2, 500)

        if not responses:
            sys.exit(
                "Unable to confirm command "
                + command
                + " for "
                + self._connection.get_info().to_string()
            )
        else:
            print(self._prefix + " " + responses[0])

    def _inertial_callback(self, message) -> None:
        """
        Inertial callback that routes IMU data from two devices into
        the global variables for sensor 1 and sensor 2.

        The serial numbers are hard-coded; adjust them if your hardware differs.
        """
        global gyr_x_1, gyr_y_1, gyr_z_1
        global acc_x_1, acc_y_1, acc_z_1
        global gyr_x_2, gyr_y_2, gyr_z_2
        global acc_x_2, acc_y_2, acc_z_2

        if self._prefix == "65577B49":
            gyr_x_1 = message.gyroscope_x
            gyr_y_1 = message.gyroscope_y
            gyr_z_1 = message.gyroscope_z
            acc_x_1 = message.accelerometer_x
            acc_y_1 = message.accelerometer_y
            acc_z_1 = message.accelerometer_z
        elif self._prefix == "655782F7":
            gyr_x_2 = message.gyroscope_x
            gyr_y_2 = message.gyroscope_y
            gyr_z_2 = message.gyroscope_z
            acc_x_2 = message.accelerometer_x
            acc_y_2 = message.accelerometer_y  # fixed bug in original code
            acc_z_2 = message.accelerometer_z


def check_root_directory(root_directory: str) -> None:
    """
    Ensure ROOT_DIRECTORY exists and handle any existing content.
    """
    if not os.path.exists(root_directory):
        os.makedirs(root_directory)
        return

    if not os.listdir(root_directory):
        return

    if ask_yn(f"\033c{root_directory} is not empty. Do you want to clear it? (Y/N)"):
        print("Clearing existing data...")
        for folder in os.listdir(root_directory):
            folder_path = os.path.join(root_directory, folder)
            if not os.path.isdir(folder_path):
                continue
            for filename in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, filename))
            os.rmdir(folder_path)
    elif ask_yn("Do you want to archive it by renaming the folder? (Y/N)"):
        folder_name = str(input("New folder name: ")).strip()
        if folder_name and root_directory != folder_name:
            os.rename(root_directory, folder_name)
            os.makedirs(root_directory, exist_ok=True)
        else:
            sys.exit("Invalid folder name. Aborting.")
    else:
        sys.exit("Cannot write into a non-empty folder. Exiting.")


def check_wifi(expected_ssid: str) -> None:
    """
    Validate that the machine is connected to the expected Wi-Fi network.

    If expected_ssid is empty, the check is skipped.
    """
    if not expected_ssid:
        return

    print("Checking Wi-Fi ...")
    try:
        connected_ok, ssid = connected_wifi()
    except Exception:
        print("Wi-Fi check function failed; continuing without enforcing SSID.")
        return

    if connected_ok:
        if ssid not in {expected_ssid, expected_ssid + "_5G"}:
            print(f"Warning: not connected to expected Wi-Fi ({expected_ssid}). Current: {ssid}")
        else:
            print(LINE_UP, end=LINE_CLEAR)
            print(f"Connected to {ssid}")
    else:
        print("Could not check Wi-Fi")


def establish_imu_connections():
    """
    Discover and establish UDP connections to all available XIMU3 devices.
    """
    print("Checking connection to IMUs ...")
    connections = []
    while True:
        try:
            messages = ximu3.NetworkAnnouncement().get_messages_after_short_delay()
            connections = [Connection(m.to_udp_connection_info()) for m in messages]
            break
        except AssertionError:
            continue

    if not connections:
        print(LINE_UP, end=LINE_CLEAR)
        sys.exit("No UDP connections to IMUs")

    print(LINE_UP, end=LINE_CLEAR)
    print("Connected to IMUs")
    return connections


def print_imu_status_message(clean_folder: bool) -> str:
    """
    Build and print the status header message.
    """
    message = (
        "Programme running   ctrl + C to stop\n\n"
        f"Clean Folder on exit: {clean_folder}\n"
    )
    print("\033c" + message)
    return message


def clean_buffer(root_directory: str, sample_counter: int, buffer_size: int) -> None:
    """
    Remove old sample folders once the rolling buffer size is exceeded.
    """
    target_sample = sample_counter - buffer_size
    if target_sample <= 0:
        return

    sample_dir = os.path.join(root_directory, f"Sample_{target_sample}")
    if not os.path.exists(sample_dir):
        return

    for filename in os.listdir(sample_dir):
        os.remove(os.path.join(sample_dir, filename))
    os.rmdir(sample_dir)


def main() -> None:
    print("\033cStarting ...\n")

    # Directory check
    check_root_directory(ROOT_DIRECTORY)

    # Wi-Fi check
    check_wifi(WIFI_TO_CONNECT)

    # IMU connections
    connections = establish_imu_connections()

    # Countdown to start
    try:
        input("\nProgramme Ready, Press Enter to Start")
        for i in range(2, 0, -1):
            print(f"Starting in {i}s")
            sleep(1)
            print(LINE_UP, end=LINE_CLEAR)
    except KeyboardInterrupt:
        for connection in connections:
            connection.close()
        sys.exit("\nProgramme Stopped\n")

    sequence_length = 10  # number of IMU rows per sample
    sample_counter = 0
    line_counter = 0      # total IMU lines written
    csv_file = None
    start_time = time()

    try:
        status_message = print_imu_status_message(CLEAN_FOLDER)

        while True:
            sample_counter += 1

            # Create sample folder and CSV
            sample_dir = os.path.join(ROOT_DIRECTORY, f"Sample_{sample_counter}")
            os.makedirs(sample_dir)
            csv_path = os.path.join(sample_dir, "imu.csv")
            csv_file = open(csv_path, mode="w", newline="")
            csv_writer = csv.writer(csv_file)

            # Collect a sequence of IMU samples
            for _ in range(sequence_length):
                line_counter += 1

                # Timing control for IMU sample rate
                while time() - start_time < line_counter / FPS:
                    sleep(0.001)

                # Write IMU row
                csv_writer.writerow(
                    [
                        gyr_x_1,
                        gyr_y_1,
                        gyr_z_1,
                        acc_x_1,
                        acc_y_1,
                        acc_z_1,
                        gyr_x_2,
                        gyr_y_2,
                        gyr_z_2,
                        acc_x_2,
                        acc_y_2,
                        acc_z_2,
                    ]
                )

                # Optional IMU terminal output
                if PRINT_IMU:
                    gyr1_vals = [round(gyr_x_1), round(gyr_y_1), round(gyr_z_1)]
                    gyr2_vals = [round(gyr_x_2), round(gyr_y_2), round(gyr_z_2)]

                    tabulation = "\t" if len(str(gyr1_vals)) >= 15 else "\t\t"
                    tabulation2 = "\t" if len(str(gyr2_vals)) >= 14 else "\t\t"

                    print(gyr1_vals, tabulation, gyr2_vals, tabulation2, end="")
                    print(
                        f": Line {line_counter} "
                        f"Sample {sample_counter} "
                        f"at {round(time() - start_time, 2)}"
                    )

                    if line_counter % WINDOW_SIZE == 0:
                        print("\033c" + status_message)

            # Maintain rolling buffer of sample folders
            clean_buffer(ROOT_DIRECTORY, sample_counter, BUFFER)

            csv_file.close()
            csv_file = None

    except KeyboardInterrupt:
        elapsed = round(time() - start_time, 4)
        rate_effective = line_counter / elapsed if elapsed > 0 else 0.0
        print(
            f"\n{line_counter} IMU samples were saved in "
            f"{format_time(elapsed)}  -  effective rate: {rate_effective:.2f} Hz"
        )

        if csv_file:
            csv_file.close()

        if CLEAN_FOLDER:
            for folder in os.listdir(ROOT_DIRECTORY):
                folder_path = os.path.join(ROOT_DIRECTORY, folder)
                if not os.path.isdir(folder_path):
                    continue
                for filename in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, filename))
                os.rmdir(folder_path)
            os.rmdir(ROOT_DIRECTORY)

    finally:
        # Release IMU connections
        for connection in connections:
            connection.close()
        print("\nProgramme Stopped\n")


if __name__ == "__main__":
    main()