import os
import time
import sys
import clr
import ctypes
import tisgrabber as tis
import cv2
import numpy as np
import math
from ultralytics import YOLO
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.Benchtop.BrushlessMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.Tools.Logging.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.Tools.WPF.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.PrivateInternal.dll")
clr.AddReference("System")
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.Benchtop.BrushlessMotorCLI import *
from System import Decimal  # necessary for real world units
from System import Math


x_increment = Decimal(0.4) # how much the X-stage moves after every image in mm
y_increment = Decimal(0.4) # how much the Y-stage moves after every image in mm
camera_serial_number = "DFK 33UX264"
motor_serial_number = "73109504"
move_distance = 0.2 #how much total distance to move when scanning edges in mm
# assumes that camera is oriented so that when stage moves right, view moves right(camera is "upside down")
#Home motor, put bottom right corner of chip in view of camera - should look like top right corner on image
x_position = 0
y_position = 0
video_format = (640,480)



def main():
    """The main entry point for the application"""
    start_time = time.time()
    model = YOLO("best.pt")
    try:
        #initialize camera
        ic = ctypes.cdll.LoadLibrary("./tisgrabber_x64.dll")
        tis.declareFunctions(ic)
        ic.IC_InitLibrary(0)
        hGrabber = ic.IC_CreateGrabber()
        ic.IC_OpenVideoCaptureDevice(hGrabber, tis.T(camera_serial_number))
        if not ic.IC_IsDevValid(hGrabber):
            print("Invalid Device")
            ic.IC_ReleaseGrabber(hGrabber)
            return -1
        video_format_str = "RGB32 (" + str(video_format[0]) + "x" + str(video_format[1]) + ")"
        ic.IC_SetVideoFormat(hGrabber, tis.T(video_format_str))

        #find all available motor devices
        DeviceManagerCLI.BuildDeviceList()
        # Create device
        device = BenchtopBrushlessMotor.CreateBenchtopBrushlessMotor(motor_serial_number)
        device.Connect(motor_serial_number)
        #x channel
        channel1 = device.GetChannel(1)
        #y channel
        channel2 = device.GetChannel(2)

        # Start polling and enable
        channel1.StartPolling(250)  #250ms polling rate
        time.sleep(0.25)
        channel1.EnableDevice()
        time.sleep(0.25)  # Wait for device to enable

        channel2.StartPolling(250)  #250ms polling rate
        time.sleep(0.25)
        channel2.EnableDevice()
        time.sleep(0.25)
      
        # Get Device Information and display description
        device_info = channel1.GetDeviceInfo()
        print(device_info.Description)

        # Load any configuration settings needed by the controller/stage
        print(channel1)
        motor_config = channel1.LoadMotorConfiguration(channel1.DeviceID)  # Device ID is the serial no + channel
        device_settings = channel1.MotorDeviceSettings
        motor_config.UpdateCurrentConfiguration()
        channel1.SetSettings(device_settings, False)
        
        print(channel2)
        motor_config = channel2.LoadMotorConfiguration(channel2.DeviceID)  # Device ID is the serial no + channel
        device_settings = channel2.MotorDeviceSettings
        motor_config.UpdateCurrentConfiguration()
        channel2.SetSettings(device_settings, False)

        if not channel1.IsSettingsInitialized():
            channel1.WaitForSettingsInitialized(1000)  # 10 second timeout
            assert channel1.IsSettingsInitialized() is True
            
        if not channel2.IsSettingsInitialized():
            channel2.WaitForSettingsInitialized(1000)  # 10 second timeout
            assert channel2.IsSettingsInitialized() is True

 
        #account for unevenness in lighting
        ic.IC_StartLive(hGrabber, 1)
        kernel = np.zeros(video_format)
        take_blank = input("Press 1 to take a new image of a blank background, and Enter if blank.jpg already exists and you want to use it")
        if (take_blank == '1'):
            done = input("Place a white background in view and press Enter to continue")
            if ic.IC_SnapImage(hGrabber, 2000) == tis.IC_SUCCESS:
                ic.IC_SaveImage(hGrabber, tis.T("blank.jpg"), tis.ImageFileTypes['JPEG'],90)
        blank = cv2.imread("blank.jpg", 0)
        kernel = np.max(blank) - blank


        # Get parameters related to homing/zeroing/other
        home_params1 = channel1.GetHomingParams()
        print(f'Homing Velocity: {home_params1.Velocity}\n',
              f'Homing Direction: {home_params1.Direction}')

        # Home or Zero the device (if a motor/piezo)
        print("Homing Channel 1")
        channel1.Home(10000)
        print("Channel 1 Homed")
        
        home_params2 = channel2.GetHomingParams()
        print(f'Homing Velocity: {home_params2.Velocity}\n',
              f'Homing Direction: {home_params2.Direction}')

        # Home or Zero the device (if a motor/piezo)
        print("Homing Channel 2")
        channel2.Home(10000)
        print("Channel 2 Homed")
        
        print("Place the bottom right corner of chip in view (chip should be on the bottom right portion of the screen)")
        done = input("Press Enter when done")
        global x_position, y_position, x_increment, y_increment
        x_position = channel1.DevicePosition
        y_position = channel2.DevicePosition

        edge_positions = [[Decimal.ToDouble(x_position), Decimal.ToDouble(y_position)]]
        home_x, home_y = x_position, y_position
        #start scanning the edges of the chip
        #always moves right first as its first move (can change)
        x_position = x_position + Decimal(2 * move_distance) + Decimal(0.01) # add 0.01 so that it is out of range that will make the while loop stop
        x_inc, y_inc = Decimal(move_distance), Decimal(0)
        edge_positions.append([Decimal.ToDouble(x_position),Decimal.ToDouble(y_position)])

        channel1.MoveTo(x_position, 20000)
        #y_position += Decimal(-0.801)
        #edge_positions.append([x_position, y_position])
        #channel2.MoveTo(y_position, 20000)
        #x_inc, y_inc = Decimal(0), Decimal(-0.2)
        i = 0
        average_std = 0.0
        #go around the entire chip
        while (Decimal.ToDouble(Math.Abs(x_position - home_x)) > move_distance * 2 or Decimal.ToDouble(Math.Abs(y_position - home_y)) > move_distance * 2):
            file_name = "edge" + str(i) + ".jpg"
            image = getImageGrayscale(ic, hGrabber, kernel, file_name)
            std = np.std(image)
            print(average_std)
            # gone off the edge
            if (std < average_std/3):
                print("off")
                #go back to previous position
                channel1.MoveTo(x_position, 20000)
                channel2.MoveTo(y_position, 20000)
                image, std = search(ic, hGrabber, channel1, channel2, kernel, average_std)                    
            average_std = std * 1/(1+i) + average_std * i/(1+i)                                        
            image = center(image, ic, hGrabber, channel1, channel2, kernel)
            image = getBinary(image)
            #find next move
            x_inc, y_inc = getEdgeDirection(image, Decimal.ToDouble(x_inc), Decimal.ToDouble(y_inc))
            x_inc = Decimal(x_inc)
            y_inc = Decimal(y_inc)
            print(x_inc)
            print(y_inc)
            #update current position
            x_position += x_inc
            y_position += y_inc
            print(f"X_position: {x_position}")
            print(f"Y_position: {y_position}")
            channel1.MoveTo(x_position, 20000)
            channel2.MoveTo(y_position, 20000)
            #save current position
            edge_positions.append([float(Decimal.ToDouble(x_position)), float(Decimal.ToDouble(y_position))])
            i += 1
        edge_positions = np.array(edge_positions)
        #find boundary positions of chip from saved positions
        # assume home position is bottom left corner
        y_bound = np.min(edge_positions[:,1])
        y = Decimal.ToDouble(home_y)
        x_bounds = []
        y_range = Decimal.ToDouble(y_increment)
        #for every y value that the motor will pass through, find the range of x values it needs to go through
        while (y > y_bound):
            x_values = edge_positions[edge_positions[:,1] < y + y_range]
            x_values = x_values[x_values[:,1] > y - y_range]
            x_values = x_values[:,0]
            x_range = (np.min(x_values), np.max(x_values))
            x_bounds.append(x_range)
            y -= y_range   

        #Go back to home position in order to scan the chip and find monolayers
        i = 0
        channel1.MoveTo(home_x, 20000)
        channel2.MoveTo(home_y, 20000)
        print("Starting to scan")
        x_position = home_x
        y_position = home_y
        y_bound = Decimal(y_bound)
        loop_start = time.time()
        home_x_str = "X:" + str(Decimal.ToDouble(home_x)) + "\n"
        home_y_str = "Y:" + str(Decimal.ToDouble(home_y)) + "\n"
        # will overwrite anything existing in file
        with open("output.txt", "w") as output:
            output.write("Position of bottom right corner:\n")
            output.write(home_x_str)
            output.write(home_y_str)
            output.write("Positions of monolayers in (x,y) format:\n")
            output.close()
        #scan the chip
        while (y_position > y_bound):
            while (x_position >= Decimal((x_bounds[i])[0]) and x_position <= Decimal((x_bounds[i])[1])):
                file_name = "test.jpg"
                getImage(ic, hGrabber, file_name)
                #adjust hue on image to get cooler monolayer colors that make them easier to detect
                adjustHue(file_name)
                #run image through pretrained YOLO model  
                results = model(file_name)
                for r in results:
                    if(r.boxes.shape[0] > 0):
                        file_name = str(Decimal.ToDouble(x_position)) + "," + str(Decimal.ToDouble(y_position)) + ".jpg"
                        r.save(filename = file_name)
                        with open("output.txt", "a") as output:
                            output.write("(" + file_name[:-4] + ")\n")
                            output.close()                                           
                x_position += x_increment
                channel1.MoveTo(x_position, 20000)
            y_position -= y_increment
            #move to minimum or max x depending on what direction you're going
            if x_increment > Decimal(0) and i + 1 < len(x_bounds):
                x_position = Decimal(x_bounds[i+1][1])
            elif x_increment < Decimal(0) and i + 1 < len(x_bounds):
                x_position = Decimal(x_bounds[i+1][0])
            channel2.MoveTo(y_position, 20000)
            channel1.MoveTo(x_position, 20000)
            x_increment = -x_increment
            i += 1
                
        output.close()        
        #close camera
        ic.IC_StopLive(hGrabber)
        ic.IC_ReleaseGrabber(hGrabber)
        # Stop Polling and Disconnect
        channel1.StopPolling()
        channel2.StopPolling()
        device.Disconnect()
        end_time = time.time()
        print(str(end_time - start_time))
        print(str(end_time - loop_start))
    
    except Exception as e:
        print(e)


def getImageGrayscale(ic, hGrabber, kernel, file_name):
    if ic.IC_SnapImage(hGrabber, 2000) == tis.IC_SUCCESS:
        ic.IC_SaveImage(hGrabber, tis.T(file_name), tis.ImageFileTypes['JPEG'], 90)
        image = cv2.imread(file_name, 0)
        #remove effects of uneven lighting when finding edges
        image = image.astype(np.uint32) + kernel
        image[image > 255] = 255
        image = image.astype(np.uint8)
        return image
def getImage(ic, hGrabber, file_name):
    if ic.IC_SnapImage(hGrabber, 2000) == tis.IC_SUCCESS:
        ic.IC_SaveImage(hGrabber, tis.T(file_name), tis.ImageFileTypes['JPEG'], 90)

#used if program has lost track of the edge
def search(ic, hGrabber, channel1, channel2, kernel, average_std):
    radius = 0.1
    global x_position, y_position
    std = 0
    max_index = 0
    positions = ()
    #use standard deviation of image as a measure of whether it is on the edge
    while (std < average_std/3):
        full_move = Decimal(radius)
        half_move = Decimal(radius/math.sqrt(2))
        #store 8 positions around the current position at the given radius away
        positions = ((x_position, y_position + full_move),(x_position + half_move, y_position + half_move), (x_position + full_move, y_position), (x_position + half_move, y_position - half_move), (x_position, y_position - full_move),
        (x_position - half_move, y_position - half_move), (x_position - full_move, y_position), (x_position - half_move, y_position + half_move))
        stds = []
        for position in positions:
            channel1.MoveTo(position[0], 20000)
            channel2.MoveTo(position[1], 20000)
            time.sleep(1)
            stds.append(np.std(getImageGrayscale(ic, hGrabber, kernel, "temp.jpg")))
        print(stds)
        max_index = np.argmax(np.array(stds))
        #see if the edge was found in one of these positions
        std = stds[max_index]
        #go back to original position
        channel1.MoveTo(x_position, 20000)
        channel2.MoveTo(y_position, 20000)
        radius += 0.1
    #go to the found edge
    x_position = positions[max_index][0]
    y_position = positions[max_index][1]
    channel1.MoveTo(x_position, 20000)
    channel2.MoveTo(y_position, 20000)
    return getImageGrayscale(ic, hGrabber, kernel, "temp.jpg"), std

#gets a crude binary image by blurring and thresholding
def preProcess(image):
    width = image.shape[1]
    height = image.shape[0]
    image = cv2.blur(image, (100,100))
    lowest = np.min(image)
    contrast = np.max(image) - lowest
    threshold = lowest + contrast/4
    image[image > threshold ] = 255
    image [image < threshold] = 0
    return image
def getBinary(image):
    width = image.shape[1]
    height = image.shape[0]
    blurred = cv2.blur(image, (130,130))
    blurred = cv2.blur(image, (160,160))
    lowest = np.min(blurred)
    contrast = np.max(blurred) - np.min(blurred)
    threshold = lowest + contrast/4
    blurred[blurred >= threshold] = 255
    blurred[blurred < threshold] = 0
    blurred = cv2.blur(blurred, (100,100))
    mean = np.mean(blurred)
    blurred[blurred >= mean] = 255
    blurred[blurred < mean] = 0
    return blurred

#given an image and the previous move of the motor, calculate the next move to stay along the edge
def getEdgeDirection(image, previous_x, previous_y):
    cv2.imwrite("binary.jpg", image)
    differences_y = np.diff(image, axis = 1)# checks for vertical borders
    indices_y = np.argwhere(differences_y != 0)
    rotated = np.rot90(image)
    differences_x = np.diff(rotated, axis = 1)
    indices_x = np.argwhere(differences_x !=0) # checks for horizontal borders
    #choose which one to use based on whether border is more vertical or horizontal
    if (indices_y[:,1].size > indices_x[:,1].size):
        rows = indices_y[:,0] # rows where a change occurred
        same_row = np.diff(rows) #yields zero when there are two indices in the same row (unwanted)
        mask = (same_row == 0) 
        remove_indices = np.nonzero(mask)[0] # finds the indices of repeat rows
        if (remove_indices.size > 0):
            if (remove_indices[0] > 0): #if the topmost row only has one border, use that as the guide
                # for each row that is repeated, choose only one value
                # value is determined by the row above that only has one switch from black to white
                #ultimately, the first row out of two repeated ones is deleted, so only the second row needs to have the correct value
                for row in remove_indices: 
                    above = indices_y[row - 1][1]
                    current = indices_y[row][1]
                    below = indices_y[row + 1][1]
                    if (abs(current - above) < abs(below - above)):
                        indices_y[row + 1][1] = current
            #if the first row is not usable because there are two switches, start from the bottom instead
            elif (remove_indices[-1] < rows[-1]):
                for row in remove_indices:
                    current = indices_y[row][1]
                    below = indices_y[row + 1][1]
                    under = indices_y[row + 1][1]
                    if (abs(current - under) < abs(below - under)):
                        indices_y[row + 1][1] = current
            # now only use rows that appear once
            mask = np.invert(mask)
            #np.diff decreases array size by one, need to put it back
            if (mask[-1] == False):
                mask = np.append(mask, False)
            else:
                mask = np.append(mask, True)
            # get the original indices back, with only one edge running along the image
            indices_y = indices_y[mask]
        indices_y = indices_y[:, 1] #columns of each row where the pixel changes value
        slopes_y = np.diff(indices_y)# for every row, how many pixels the column changes along the edge
        #take the mean of every 10 differences to get a slope value (if the slope is less than 1/10th, it will appear horizontal)
        last = slopes_y.size//10
        leftover = slopes_y[last * 10: -1]
        slopes_y = np.reshape(slopes_y[0:last * 10], (last, 10))
        slopes_y = np.mean(slopes_y, axis = 1)
        if (leftover.size > 0):
            leftover = np.mean(leftover)
            np.append(slopes_y, leftover)
        #bin slopes by angle, take the median
        angles = np.arctan(slopes_y)
        bins = np.linspace(-np.pi/2, np.pi/2, num = 91, endpoint = True)
        frequencies = np.digitize(angles, bins)
        #mode = stats.mode(frequencies)[0]
        median = int(np.median(frequencies))
        #get the middle angle value of the bin
        angle = bins[median]       
        slope_y = math.tan(angle - math.radians(1))# how many vertical columns you go right for every row you go down
        total = math.sqrt(1 + slope_y**2)
        #the coordinates the motor should move, normalized to move_distance
        x, y = slope_y * move_distance/total, 1 * move_distance/total
        # pass in -y because positive y on an image is downward(-y on motor)
        return clockwise(x,-y, previous_x, previous_y)

    #use rotated image if slope is horizontal - hard to distinguish otherwise
    else:
        rows = indices_x[:,0]
        same_row = np.diff(rows)
        mask = (same_row == 0)
        remove_indices = np.nonzero(mask)[0]
        if (remove_indices.size > 0):
            if (remove_indices[0] > 0):
                for row in remove_indices: 
                    above = indices_x[row - 1][1]
                    current = indices_x[row][1]
                    below = indices_x[row + 1][1]
                    if (abs(current - above) < abs(below - above)):
                        indices_x[row + 1][1] = current
            elif (remove_indices[-1] < rows[-1]):
                for row in remove_indices:
                    current = indices_x[row][1]
                    below = indices_x[row + 1][1]
                    under = indices_x[row + 1][1]
                    if (abs(current - under) < abs(below - under)):
                        indices_x[row + 1][1] = current
            mask = np.invert(mask)
            if (mask[-1] == False):
                mask = np.append(mask, False)
            else:
                mask = np.append(mask, True)
            indices_x = indices_x[mask]
        indices_x = indices_x[:, 1]
        #print(indices_x)
        slopes_x = np.diff(indices_x)
        last = slopes_x.size//10
        leftover = slopes_x[last * 10: -1]
        slopes_x = np.reshape(slopes_x[0:last * 10], (last, 10))
        slopes_x = np.mean(slopes_x, axis = 1)
        if leftover.size > 0:
            leftover = np.mean(leftover)
            np.append(slopes_x, leftover)
        angles = np.arctan(slopes_x)
        bins = np.linspace(-np.pi/2, np.pi/2, num = 91, endpoint = True)
        frequencies = np.digitize(angles, bins)
        #mode = stats.mode(frequencies)[0]
        median = int(np.median(frequencies))
        angle = bins[median]
        slope_x = math.tan(angle - math.radians(1)) # how many horizontal rows you go down for every column you go left
        total = math.sqrt(1 + slope_x **2)
        x, y = 1 * move_distance/total, slope_x * move_distance/total
        print(x)
        print(y)
        # -x, -y because x_slope measures down and left
        return clockwise(-x, -y, previous_x, previous_y)
def angleBetween(vector1, vector2):
    cos = (vector1[0]*vector2[0] + vector1[1]*vector2[1])/(move_distance**2)
    if (cos > 1 and cos < 1.01):
        return 0
    if (cos < -1 and cos > -1.01):
        return np.pi
    else:
        return math.acos(cos)
def clockwise(x,y, previous_x, previous_y):
        angle1 = angleBetween((x,y),(previous_x,previous_y))
        angle2 = angleBetween((-x,-y),(previous_x,previous_y))
        angle = min(angle1, angle2)
        #a large angle change indicates some type of corner
        if (angle > math.pi/4):
            #choose move that will turn clockwise
            if (previous_x*y - x*previous_y < 0):
                return x,y
            else:
                return -x, -y
        #otherwise, just choose the one that deviates the least from the previous move
        if (angle == angle1):
            return x,y
        else:
            return -x,-y

#centers the edge on an image
def center(image, ic, hGrabber, channel1, channel2, kernel):
    global x_position, y_position
    image = preProcess(image)
    mean = np.mean(image)
    width = image.shape[1]
    height = image.shape[0]
    if (mean < 95):
        top = np.mean(image[0:int(height/2),:])
        bottom = np.mean(image[int(height/2):height,:])
        left = np.mean(image[:, 0:int(width/2)])
        right = np.mean(image[:,int(width/2):width])
        if (top - bottom > 35):
            y_position += Decimal(0.05)
            channel2.MoveTo(y_position, 20000)
        elif (bottom - top > 35):
            y_position -= Decimal(0.05)
            channel2.MoveTo(y_position, 20000)
        if (left - right > 35):
            x_position -= Decimal(0.05)
            channel1.MoveTo(x_position, 20000)
        elif (right - left > 35):
            x_position += Decimal(0.05)
            channel1.MoveTo(x_position, 20000)
        image = getImageGrayscale(ic, hGrabber, kernel, "new.jpg")
        image = preProcess(image)
        mean = np.mean(image)
    if (mean > 160):
        top = np.mean(image[0:int(height/2),:])
        bottom = np.mean(image[int(height/2):height,:])
        left = np.mean(image[:, 0:int(width/2)])
        right = np.mean(image[:,int(width/2):width])
        if (top - bottom > 35):
            y_position -= Decimal(0.05)
            channel2.MoveTo(y_position, 20000)
        elif (bottom - top > 35):
            y_position += Decimal(0.05)
            channel2.MoveTo(y_position, 20000)
        if (left - right > 35):
            x_position += Decimal(0.05)
            channel1.MoveTo(x_position, 20000)
        elif (right - left > 35):
            x_position -= Decimal(0.05)
            channel1.MoveTo(x_position, 20000)
        image = getImageGrayscale(ic, hGrabber, kernel, "new.jpg")
        image = preProcess(image)
        mean = np.mean(image)
    return image

#assumes monolayers appear teal
#adjusts the hue to make them more purple - increases contrast
def adjustHue(file_name):
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image.astype(np.uint32)
    #rotates the hue channel by 19/180
    image[:,:,0] += 19
    image[:,:,0] = image[:,:,0]%180
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(file_name, image)

if __name__ == "__main__":
    main()

