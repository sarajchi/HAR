if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

# ----   # Modifiable variables   ----
root_directory: str =   'Temporary_Data'    # Directory where temporary folders are stored
Ask_cam_num: bool =     False               # Set to True to ask the user to put the cam number themselves, if False, default is set below
cam_num: int =          0                   # Set to 0 to activate the camera, but 1 if yoy have a builtin camera
NEW_CAM : bool =        False               # Set to True if you are using the new camera
fps: int =              30                  # Number of save per seconds
buffer: int =           1500                # Number of folders saved
CleanFolder: bool =     False               # If True, delete all temporary folders at the end
wifi_to_connect: str =  'Upper_Limb_Exo'    # The Wi-Fi where the raspberry pi and IMUs are connected
window_size: int =      200                  # How many lines of IMU data will be displayed at the same time
PRINT_IMU =             True                # If true print the imu data in the terminal
# ------------------------------------
 
import csv                      # For csv writing
import os                       # To manage folders and paths
import sys                      # For quitting program early
from time import sleep, time    # To get time and wait

try :
    import cv2      # For the camera
    import ximu3    # For the IMU
    from pupil_labs.realtime_api.simple import discover_one_device
    import pandas as pd
except ModuleNotFoundError as Err :
    missing_module = str(Err).replace('No module named ','')
    missing_module = missing_module.replace("'",'')
    if missing_module == 'cv2' :
        sys.exit(f'No module named {missing_module} try : pip install opencv-python')
    elif missing_module == "pupil_labs" :
        sys.exit(f'No module named {missing_module} try : pip install pupil-labs-realtime-api')
    else :
        print(f'No module named {missing_module} try : pip install {missing_module}')

try :
    from Imports.Functions import format_time, connected_wifi, ask_yn
except ModuleNotFoundError :
    sys.exit('Missing Import folder, make sure you are in the right directory')

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

# Initialize sensor values to 0
gyr_x_1 = gyr_y_1 = gyr_z_1 = 0
acc_x_1 = acc_y_1 = acc_z_1 = 0
gyr_x_2 = gyr_y_2 = gyr_z_2 = 0
acc_x_2 = acc_y_2 = acc_z_2 = 0

try :
    if Ask_cam_num :
        NEW_CAM = ask_yn('\033cAre you using the new camera ?(Y/N) ')
        if  not NEW_CAM:
            cam_num = int(input("Cam Number : "))
        if cam_num < 0: 
            raise ValueError
except (ValueError, TypeError) :
    sys.exit("Invalid Cam Number") 
except KeyboardInterrupt :
    sys.exit("\n\nProgramme Stopped\n")

# We check if the root directory exist
if not os.path.exists(root_directory) :
    os.makedirs(root_directory)
elif os.listdir(root_directory):  # If there are files in the directory : True
    if ask_yn(f'\033c{root_directory} not empty do you want to clear it ? (Y/N)') :
        print('Clearing ...')
        for folders_to_del in os.listdir(root_directory):
            for files_to_del in os.listdir(f"{root_directory}/{folders_to_del}"):
                os.remove(os.path.join(f'{root_directory}/{folders_to_del}', files_to_del))
            os.rmdir(f"{root_directory}/{folders_to_del}")
    elif ask_yn('Do you want to save it ? (Y/N)') :
        Folder_Name = str(input("Folder Name : "))
        if root_directory != Folder_Name and Folder_Name != '' :
            os.rename(root_directory, Folder_Name)
        else : sys.exit("Incorrect Folder Name")
    else : sys.exit('Cannot access non-empty folder, Programme Stopped\n')

print("\033cStarting ...\n") # Clear Terminal
print("Checking Wifi ...")

ConnectedWifi = connected_wifi()
if ConnectedWifi[0] :
    if ConnectedWifi[1] != wifi_to_connect and ConnectedWifi[1] != wifi_to_connect+'_5G' :
        sys.exit('Not connected to the right wifi')
    else : 
        print(LINE_UP, end=LINE_CLEAR)
        print(f'Connected to {ConnectedWifi[1]}')
else : print("Could not check Wifi")


class Connection:
    def __init__(self, connection_info):
        self.__connection = ximu3.Connection(connection_info)

        if self.__connection.open() != ximu3.RESULT_OK:
            sys.exit("Unable to open connection " + connection_info.to_string())

        ping_response = self.__connection.ping()

        if ping_response.result != ximu3.RESULT_OK:
            print("Ping failed for " + connection_info.to_string())
            raise AssertionError

        self.__prefix = ping_response.serial_number
        self.__connection.add_inertial_callback(self.__inertial_callback)

    def close(self):
        self.__connection.close()

    def send_command(self, key, value=None):
        if value is None:
            value = "null"
        elif type(value) is bool:
            value = str(value).lower()
        elif type(value) is str:
            value = "\"" + value + "\""
        else:
            value = str(value)

        command = "{\"" + key + "\":" + value + "}"

        responses = self.__connection.send_commands([command], 2, 500)

        if not responses:
            sys.exit("Unable to confirm command " + command + " for " + self.__connection.get_info().to_string())
        else:
            print(self.__prefix + " " + responses[0])

    def __inertial_callback(self, message):
        global gyr_x_1, gyr_y_1, gyr_z_1
        global acc_x_1, acc_y_1, acc_z_1
        global gyr_x_2, gyr_y_2, gyr_z_2
        global acc_x_2, acc_y_2, acc_z_2
        if self.__prefix == '65577B49':
            gyr_x_1 = message.gyroscope_x
            gyr_y_1 = message.gyroscope_y
            gyr_z_1 = message.gyroscope_z
            acc_x_1 = message.accelerometer_x
            acc_y_1 = message.accelerometer_y
            acc_z_1 = message.accelerometer_z
        elif self.__prefix == '655782F7':
            gyr_x_2 = message.gyroscope_x
            gyr_y_2 = message.gyroscope_y
            gyr_z_2 = message.gyroscope_z
            acc_x_2 = message.accelerometer_x
            acc_y_2 = message.accelerometer_y
            acc_z_2 = message.accelerometer_z


# Establish connections
print("Checking connection to IMU ...")
while True :
    try :
        connections = [Connection(m.to_udp_connection_info()) for m in ximu3.NetworkAnnouncement().get_messages_after_short_delay()]
        break
    except AssertionError:
        pass
if not connections:
    print(LINE_UP, end=LINE_CLEAR)
    sys.exit("No UDP connections to IMUs")
print(LINE_UP, end=LINE_CLEAR)
print('Connected to IMUs')

sequence_length = 10    # Size of samples default 10
sample_counter = 0
frames_counter = 0

# Video capture setup
print("Checking camera ...")
if NEW_CAM:
    # Look for devices. Returns as soon as it has found the first device.
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        print(LINE_UP, end=LINE_CLEAR)
        sys.exit("No device found.")

    print(LINE_UP, end=LINE_CLEAR)
    print(f"Connected to {device}")

    cam_message = 'Using New Camera \n'

else :
    cap = cv2.VideoCapture(cam_num)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ret, frame = cap.read()
    if not ret: # If camera is unavailable :
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        for connection in connections:
            connection.close()
        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)
        sys.exit('Camera disconnected.')
    cam_message = f'Camera Number : {cam_num} \n'

print(LINE_UP, end=LINE_CLEAR)
print('Connected to Camera')

try :
    input('\nProgramme Ready, Press Enter to Start')
    for i in range(2) :
        print(f'Starting in {2-i}s')
        sleep(1)
        print(LINE_UP, end=LINE_CLEAR)
except KeyboardInterrupt :
    sys.exit('\nProgramme Stopped\n')

Start_Time = time()
try : # try except is to ignore the keyboard interrupt error
    message = f'Programme running   ctrl + C to stop\n\nClean Folder : {CleanFolder} \n' + cam_message
    print('\033c'+message)
    while True : # While True is an infinite loop
        sample_counter += 1

        # We create a folder with a csv file in it
        os.makedirs(f"{root_directory}/Sample_{sample_counter}")
        csv_file = open(f'{root_directory}/Sample_{sample_counter}/imu.csv', mode='w', newline='')

        # We add 1 imu data to the csv and 1 image to the folder
        for i in range(sequence_length):
            frames_counter += 1
            while time() - Start_Time < frames_counter / fps:  # To ensure right fps
                sleep(0.001)              

            # Add IMU data
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([gyr_x_1, gyr_y_1, gyr_z_1, acc_x_1, acc_y_1, acc_z_1, 
                                 gyr_x_2, gyr_y_2, gyr_z_2, acc_x_2, acc_y_2, acc_z_2])

            if NEW_CAM :
                # ret, frame = cap.read() 
                bgr_pixels, frame_datetime = device.receive_scene_video_frame()
                # ret = 
                frame = bgr_pixels # TODO Possible source of error, check conversion
                """
                if not ret: # If camera is unavailable :
                    # Release resources
                    cap.release()
                    cv2.destroyAllWindows()
                    csv_file.close()
                    for connection in connections:
                        connection.close()
                    print('\nCamera disconnected')
                    raise KeyboardInterrupt
                """
            else :
                ret, frame = cap.read()
                if not ret: # If camera is unavailable :
                    # Release resources
                    cap.release()
                    cv2.destroyAllWindows()
                    csv_file.close()
                    for connection in connections:
                        connection.close()
                    print('\nCamera disconnected')
                    raise KeyboardInterrupt
            
            
            if PRINT_IMU :
                gyr1_vals = [round(gyr_x_1), round(gyr_y_1), round(gyr_z_1)]
                gyr2_vals = [round(gyr_x_2), round(gyr_y_2), round(gyr_z_2)]

                

                if len(str(gyr1_vals)) >= 15 : tabulation = '\t'
                else : tabulation = '\t\t'
                if len(str(gyr2_vals)) >= 14 : tabulation2 = '\t'
                else : tabulation2 = '\t\t'
                print(gyr1_vals, tabulation,[round(gyr_x_2), round(gyr_y_2), round(gyr_z_2)], tabulation2, end='')
            
                print(f': Frame {frames_counter} Sample {sample_counter} at {round(time()-Start_Time,2)}')
            
                if frames_counter%window_size == 0 :
                    print('\033c'+message)

            # Add image
            image_filename = f'{root_directory}/Sample_{sample_counter}/frame_{frames_counter}.jpg'
            cv2.imwrite(image_filename, frame)

        # We delete the folders as we go so that we don't saturate
        if sample_counter > buffer:
            for files_to_del in os.listdir(f"{root_directory}/Sample_{sample_counter - buffer}"):
                os.remove(os.path.join(f'{root_directory}/Sample_{sample_counter - buffer}', files_to_del))
            os.rmdir(f"{root_directory}/Sample_{sample_counter - buffer}")





except KeyboardInterrupt :
    t = round(time() - Start_Time, 4)
    print(f"\n{frames_counter} images were saved in {format_time(t)}  -  fps : {frames_counter / t}")

    try : # We use try because csv_file can be undefined
        csv_file.close() 
    except NameError:
        pass

    if CleanFolder:
        for folders_left in os.listdir(root_directory) :
            for files_left in os.listdir(f"{root_directory}/{folders_left}"):
                os.remove(os.path.join(f'{root_directory}/{folders_left}', files_left))
            os.rmdir(f"{root_directory}/{folders_left}")
        os.rmdir(root_directory)


# Release resources
csv_file.close()
if NEW_CAM : device.close()  # explicitly stop auto-update
cv2.destroyAllWindows()
for connection in connections:
    connection.close()





if __name__ == "__main__" :
    print('\nProgramme Stopped\n')