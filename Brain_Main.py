import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import math

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX


# ///////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////Prediction//////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
def Distance_to_target(xpixels):
    xref = 73
    k = 10 * xref
    # k is our rate of change which, if it's linearly changing, should show an inverse proportion
    DTT = (k / xpixels)
    # ddt gives distance to target in ft using a refranced number of pixels at 10ft.
    return DTT


def Length_of_frame(feet_to_target):
    LOF = feet_to_target * 2
    return LOF


def Velocity_and_ratio_prediction(xlength1, xlength2, xcentriod1, xcentroid2, ycentroid1, ycentroid2):
    # time=1seconds for predections!
    number_of_x_pixels = 640
    D1 = Distance_to_target(xlength1)
    D2 = Distance_to_target(xlength2)
    # D1 and D2 represent the distance to the target in each frame
    ft_per_pixel_ratio_1 = Length_of_frame(D1) / number_of_x_pixels
    # print("frame 1 has a foot per pixel ratio of ",ft_per_pixel_ratio_1)
    ft_per_pixel_ratio_2 = Length_of_frame(D2) / number_of_x_pixels
    # print("frame 1 has a foot per pixel ratio of ",ft_per_pixel_ratio_2)
    # ft_per_pixel_ratio_1 and 2 represent the foot per pixel ratio for each frame
    LOF1 = Length_of_frame(D1)
    # print("frame 1 has a length of frame of ",LOF1)
    LOF2 = Length_of_frame(D2)
    # print("frame 2 has a length of frame of ",LOF2)
    # LOF1 and 2 get the total length of the frame for each frame
    Distance_within_frame1 = ft_per_pixel_ratio_1 * xcentriod1
    # print("the distance from the left side of frame 1 of the total length of the frame is",Distance_within_frame1)
    Distance_within_frame2 = ft_per_pixel_ratio_2 * xcentroid2
    # print("the distance from the left side of frame 2 of the total length of the frame is",Distance_within_frame2)
    # Distance_within_frame1 and 2 tell us how far in feet the target is from the left side
    Percent_of_frame1 = Distance_within_frame1 / LOF1
    # print("the percent of the frame that is covered in frame 1 is",Percent_of_frame1)
    Percent_of_frame2 = Distance_within_frame2 / LOF2
    # print("the percent of the frame that is covered in frame 2 is",Percent_of_frame2)
    # Percent_of_frame1 and 2 tell us the percentage of the total frame the target has covered
    Percentage_of_LOFs_moved = Percent_of_frame2 - Percent_of_frame1
    Percentage_of_LOFs_moved_per_second = Percentage_of_LOFs_moved / .2
    # print("the percentage of the total LOF in .2 sec is ",Percentage_of_LOFs_moved_per_second)
    # Percentage_of_LOFs_moved_per_second shows us what percentage of any total LOF the target has move over .2 seconds
    Total_percentage_moved = 1 * Percentage_of_LOFs_moved_per_second
    # total percentage moved shows us how much of the frame will be covered in 1 second
    Percent_of_predicted_frame_covered = Percent_of_frame2 + Total_percentage_moved
    # print("the total percentage of the frame for the predicted distance is", Percent_of_predicted_frame_covered)
    # Percent_of_predicted_frame_covered tells us what percentage of the frame will be covered

    Df = D2 - D1
    # print(D1,"feet away in first frame")
    # print(D2,"feet away in second frame")
    # print(Df,"foot difference between frames for .2 seconds time difference")
    # Df gives us a change in distance to the target using 2 frames .2 seconds apart

    velocity_length = Df / .2
    # print("velocity of target is",velocity_length,"feet per second")
    # velocity length tells us the speed at which the target is moving to or from the camera
    predicted_length_moved = velocity_length * 1
    # print(predicted_length_moved,"feet moved after 1 second")
    # Predicted_length_moved shows us how many feet the target will move from its last postion after 1 second
    Predicted_length_to_target = D2 + predicted_length_moved
    # print(Predicted_length_to_target,"feet is distance to target after 1 second")
    # predicted length to target adds the length moved (pos or neg) to the last known distance to target (D2)
    Predicted_length_of_frame = Length_of_frame(Predicted_length_to_target)
    # print("predicted length of the frame is ",Predicted_length_of_frame, " feet from left to right" )
    Distance_within_frame_predicted = Predicted_length_of_frame * Percent_of_predicted_frame_covered
    # print("distance within the frame is ",Distance_within_frame_predicted)
    # Distance_within_frame_predicted tells us in feet how far from the left side of the frame the target is
    Ratio_for_frame_conversion_to_pixels = Distance_within_frame_predicted / Predicted_length_of_frame
    Predicted_xcentroid = int(Ratio_for_frame_conversion_to_pixels * number_of_x_pixels)

    Change_in_ycentroid = ycentroid2 - ycentroid1
    Predicted_change_in_ycentroid = Change_in_ycentroid * 1
    Predicted_ycentroid = int(ycentroid2 + Predicted_change_in_ycentroid)

    if (Predicted_ycentroid > 480):
        Predicted_ycentroid = 470
    elif (Predicted_ycentroid < 5):
        Predicted_ycentroid = 10
    if (Predicted_xcentroid > 640):
        Predicted_xcentroid = 630
    elif (Predicted_xcentroid < 5):
        Predicted_xcentroid = 5
    return int(Predicted_xcentroid), int(Predicted_ycentroid)


# ///////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////Servo//////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
def gpiosetup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(8, GPIO.OUT)  # Servo 1
    GPIO.setup(10, GPIO.OUT)  # Servo 2
    GPIO.setup(3, GPIO.IN)  # Limit switch 1
    GPIO.setup(5, GPIO.IN)  # Limit switch 2
    # You can umcomment this if you dont like the errors at the end
    # GPIO.setwarnings(False)
    return 0;


# Time varible xy coordinates based on rpm of servo
def xy(xcoord, ycoord):
    SERVO_MOTION_DONE = False
    print("Servo Moving Starts:")
    x_c = xcoord
    y_c = ycoord

    # Define servo parameters
    x = GPIO.PWM(8, 50)
    y = GPIO.PWM(10, 50)

    # Set up default values (Change these to modify turn time and FOV)
    # from website
    # Angle of View: 54 x 41 degrees
    # Field of View: 2.0 x 1.33 m at 2 m
    rpmy = 90  # RPM of the y servo
    rpmx = 90  # RPM of the x servo
    height = 480  # FOV height in pixels
    width = 640  # FOV width in pixels
    fovheight = 60  # FOV height in degrees
    fovwidth = 90  # FOV width in degrees
    ontime = 1  # Precentage of time servo runs at slow speed (May cause inaccuries)

    # Define servo speeds (% duty cycle of 100)
    deadset = 7
    full_cw = 1
    full_ccw = 8

    up = False
    down = False
    left = False
    right = False

    # Start servo with no movement (dead set time)
    x.start(deadset)
    y.start(deadset)

    starty = height / 2
    startx = width / 2
    turntimeperpixely = 1 / ((height / fovheight) * ((rpmy * 360) / 60))
    turntimeperpixelx = 1 / ((width / fovwidth) * ((rpmx * 360) / 60))

    # Input validation
    if (y_c > height):
        y_c = height
    elif (y_c < 0):
        y_c = 0

    if (x_c > width):
        x_c = width
    elif (x_c < 0):
        x_c = 0

    # Check wich way to turn and the amount of time to turn
    testy = starty - y_c
    testx = startx - x_c
    turny = 0
    turnx = 0
    # Assuming bottom left is 0,0
    if (testy < 0):
        up = True
        down = False
    else:
        down = True
        up = False
    turny = abs(testy) * turntimeperpixely

    if (testx < 0):
        right = True
        left = False
    else:
        left = True
        right = False
    turnx = abs(testx) * turntimeperpixelx

    # Turn the servo
    if (up == True):
        y.start(full_cw)
        time.sleep(turny * ontime)
        y.start(deadset)
    elif (down == True):
        y.start(full_ccw)
        time.sleep(turny * ontime)
        y.start(deadset)
    else:
        print("Err: incorret servo info (y)")

    if (right == True):
        x.start(full_cw)
        time.sleep(turnx * ontime)
        x.start(deadset)
    elif (left == True):
        x.start(full_ccw)
        time.sleep(turnx * ontime)
        x.start(deadset)
    else:
        print("Err: incorret servo info (x)")
    SERVO_MOTION_DONE = False


def servo_calibration():
    x = GPIO.PWM(8, 50)
    y = GPIO.PWM(10, 50)

    rpmy = 90  # RPM of the y servo
    rpmx = 90  # RPM of the x servo
    xls_pos = 90  # Position of the limit switch in deg away from center
    yls_pos = 30  # Position of the limit switch in deg away from center

    turntimeperdegy = 1 / ((rpmy * 360) / 60)
    turntimeperdegx = 1 / ((rpmx * 360) / 60)
    turntimey = yls_pos * turntimeperdegy
    turntimex = xls_pos * turntimeperdegx

    deadset = 7
    full_ccw = 8
    full_cw = 1
    check_time = .05
    ontime = 1

    xcal = False
    ycal = False

    # Assuming limit switch is set to the right for x and below for y
    while (xcal != True & ycal != True):

        # Checks if the button is pressed or if it has been calibrated already for x
        if (xcal != True):
            x.start(full_cw)
            time.sleep(check_time)
            x.start(deadset)
        elif (GPIO.input(3) == 1):
            x.start(full_ccw)
            time.sleep(turntimex * ontime)
            x.start(deadset)
            xcal = True

        # Checks if the button is pressed or if it has been calibrated already for y
        if (ycal != True):
            y.start(full_ccw)
            time.sleep(check_time)
            y.start(deadset)
        elif (GPIO.input(5) == 1):
            y.start(full_cw)
            time.sleep(turntimey * ontime)
            y.start(deadset)
            ycal = True

    return;


# ///////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////VIDEO//////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def nothing(x):
    # any operation
    pass


# ///////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////MAIN //////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
def main():
    # ////////////////////////Global Trackers
    Target_X = 0
    Target_Y = 0
    Servo_X = 0
    Servo_Y = 0
    frame_counter = 0
    P_width = 0  # P_"something" is for prediction variables
    P_cx = 0
    P_cy = 0
    predX = 0
    predY = 0
    # ///////////////////////SERVO SETUP///////////////////////////////////////////////////////
    gpiosetup()
    servo_calibration()

    # ///////////////CAMERA_VIDEO_SETUP////////////////////////////////////////////////////////
    cv2.namedWindow("Trackbars")
    cv2.namedWindow("CAM FEED")
    # cv2.namedWindow("MASK")
    cv2.resizeWindow("Trackbars", 300, 200)

    cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
    cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
    cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("MIN Area", "Trackbars", 1000, 30000, nothing)
    cv2.createTrackbar("MAX Area", "Trackbars", 10000, 90000, nothing)
    cv2.createTrackbar("MODE(1/2)", "Trackbars", 0, 1, nothing)
    cv2.createTrackbar("FIRE", "Trackbars", 0, 1, nothing)

    scale = 0.8  # window scale
    shoot_wait = 0
    SERVO_MOTION_DONE = True
    # //////////////////////////////////////////////////////////////////////

    while True:

        Switch_Track = int(cv2.getTrackbarPos("MODE(1/2)", "Trackbars"))
        Switch_Fire = int(cv2.getTrackbarPos("FIRE", "Trackbars"))

        _, frame = cap.read()

        # 640,480 is the image size

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L-H", "Trackbars")
        l_s = cv2.getTrackbarPos("L-S", "Trackbars")
        l_v = cv2.getTrackbarPos("L-V", "Trackbars")
        u_h = cv2.getTrackbarPos("U-H", "Trackbars")
        u_s = cv2.getTrackbarPos("U-S", "Trackbars")
        u_v = cv2.getTrackbarPos("U-V", "Trackbars")

        lower_red = np.array([l_h, l_s, l_v])  # set hsv
        upper_red = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        kernelE = np.ones((10, 10), np.uint8)
        kernelD = np.ones((6, 6), np.uint8)
        imgErode = cv2.erode(mask, kernelE)
        imgBlur = cv2.GaussianBlur(imgErode, (9, 9), 1)
        imgDil = cv2.morphologyEx(imgBlur, cv2.MORPH_OPEN, kernelD)
        imgFinalMask = imgDil

        if Switch_Track == 0:
            imgStack1 = stackImages(scale, [[frame], [imgFinalMask]])
            cv2.imshow("CAM FEED", imgStack1)
            # cv2.imshow("CAM FEED", frame)
            # cv2.imshow("MASK", imgFinalMask)

        if Switch_Track == 1:
            imgContour = frame.copy()
            contours, _ = cv2.findContours(imgFinalMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)

                areaMin = cv2.getTrackbarPos("MIN Area", "Trackbars")
                areaMax = cv2.getTrackbarPos("MAX Area", "Trackbars")
                if areaMin < area < areaMax:

                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
                    cx = 0
                    cy = 0
                    if len(approx) == 4:
                        # cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
                        x, y, w, h = cv2.boundingRect(approx)
                        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
                        x_length = x + w
                        y_length = y + h
                        M = cv2.moments(approx)
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(imgContour, (cx, cy), 7, (255, 255, 255), -1)
                        Target_X = cx
                        Target_Y = cy
                        if (frame_counter < 2):
                            P_width = w
                            P_cx = cx
                            P_cy = cy
                            frame_counter = frame_counter + 1
                        else:
                            predX, predY = Velocity_and_ratio_prediction(P_width, w, P_cx, cx, P_cy, cy)
                            frame_counter = 0
                        print(predX, predY)
                        cv2.circle(imgContour, (predX, predY), 7, (255, 0, 0), -1)
            if (SERVO_MOTION_DONE == True):
                xy(Target_X, Target_Y)  # Servo moves to position

            if Switch_Fire == 1:

                if (shoot_wait > 10):
                    print("Shooting" + str(shoot_wait))
                    shoot_wait = 0
                else:
                    shoot_wait = shoot_wait + 1

            imgStack2 = stackImages(scale, [[imgContour], [imgFinalMask]])
            cv2.imshow("CAM FEED", imgStack2)
            # cv2.imshow("CAM FEED", imgContour)
            # cv2.imshow("MASK", imgFinalMask)

        key = cv2.waitKey(5)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
