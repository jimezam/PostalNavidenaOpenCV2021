import cv2
import numpy as np
import os

##################################################################

def addImage(layer, image, scale_percent, x, y):
    """
    Add image to layer on (x,y) scaled by scale_percent.
    """
    print(image.shape)

    resized = cv2.resize(image, (
        int(image.shape[1] * scale_percent / 100),
        int(image.shape[0] * scale_percent / 100)
    ), interpolation = cv2.INTER_AREA)

    # print(resized.shape)

    h, w = resized.shape[:2]

    if(image.shape[2] != 4):
        print("Error: this image has not transparency")
        return layer

    if (x < 0 or y < 0 or
       x+w >= layer.shape[1] or y+h >= layer.shape[0]):
       print (f"Error: image outside bounds: {x}, {y}, {x+w}, {y+h} -> {layer.shape}")
       return layer

    layer[y:y+h, x:x+w] = resized

    return layer

##################################################################

def addFixedImage(layer, image, x, y, w, h):
    """
    Add image to layer on (x,y) with specified (width, height).
    """
    x_scale_percent = w * 100 / image.shape[1] 
    y_scale_percent = h * 100 / image.shape[0] 

    resized = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

    # print(resized.shape)

    h, w = resized.shape[:2]

    if(image.shape[2] != 4):
        print("Error: this image has not transparency")
        return layer

    if (x < 0 or y < 0 or
       x+w >= layer.shape[1] or y+h >= layer.shape[0]):
       print (f"Error: image outside bounds: {x}, {y}, {x+w}, {y+h} -> {layer.shape}")
       return layer
       
    layer[y:y+h, x:x+w] = resized

    return layer

##################################################################

def saveImage(image):
    """
    Save an image to file using the first available name using a template.
    """
    index = 1

    while True:
        filename = f"xmas_portrait_{index}.jpeg"

        if os.path.isfile(filename):
            index += 1
        else:
            break

    cv2.imwrite(filename, image)

    return filename

##################################################################

# Get the webcam feed

cam = cv2.VideoCapture(0)

# Get the dimensions of the webcam feed

height, width = cam.get(cv2.CAP_PROP_FRAME_HEIGHT), cam.get(cv2.CAP_PROP_FRAME_WIDTH)

# Load the cover images

tux      = cv2.imread('images/linux-santa.png', cv2.IMREAD_UNCHANGED)   # (800, 697, 4)
santa    = cv2.imread('images/santa-1.png', cv2.IMREAD_UNCHANGED)       # (800, 700, 4)
candle   = cv2.imread('images/candle-1.png', cv2.IMREAD_UNCHANGED)      # (800, 550, 4)
reindeer = cv2.imread('images/reindeer-1.png', cv2.IMREAD_UNCHANGED)    # (532, 800, 4)

# Load the hats

hat1 = cv2.imread('images/hat-1.png', cv2.IMREAD_UNCHANGED)
hat2 = cv2.imread('images/hat-2.png', cv2.IMREAD_UNCHANGED)

# Create the (empty) cover image

cover = np.zeros(shape=(int(height), int(width), 4), dtype=np.uint8)

# Add the images to the cover

addImage(cover, tux, 22, 5, 303)
addImage(cover, santa, 50, 274, 258)
addImage(cover, candle, 25, 1, 1)
addImage(cover, reindeer, 29, 400, 1)

##################################################################

# haarcascade_frontalface_default.xml         
# haarcascade_frontalface_alt.xml          
# haarcascade_frontalface_alt2.xml         
# haarcascade_frontalface_alt_tree.xml     
# haarcascade_eye_tree_eyeglasses.xml      
# haarcascade_russian_plate_number.xml
# haarcascade_eye.xml                      
# haarcascade_fullbody.xml                    
# haarcascade_smile.xml
# haarcascade_frontalcatface_extended.xml  
# haarcascade_lefteye_2splits.xml             
# haarcascade_upperbody.xml
# haarcascade_frontalcatface.xml           
# haarcascade_licence_plate_rus_16stages.xml
# haarcascade_lowerbody.xml
# haarcascade_profileface.xml
# haarcascade_righteye_2splits.xml

# Creates the HaarCascade frontal head classifier

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

##################################################################

# Process the app lifecycle

while True:
    # Read the camera feed

    ret_val, img_cam = cam.read()   

    # Mirror image

    img_cam = cv2.flip(img_cam, 1)

    # Add transparency channel to camera image

    img = cv2.cvtColor(img_cam, cv2.COLOR_RGB2RGBA)

    ##############################################################

    # Calculates the grayscale version of the image

    gray = cv2.cvtColor(img_cam, cv2.COLOR_RGB2GRAY)

    # Use the classifier to identify frontal heads in the image

    # Source image (gray in this case)
    # Scale factor (reduce the image): 1.05 (slower/better) - 1.4 (faster/worse)
    # Min neighbours is the minimal amount of neighbour frames required to hold candidate: 3-6
    # minSize smallest face dimension to be taken
    # maxSize bigest face dimension to be taken

    faces = detector.detectMultiScale(gray, 1.3, 3)

    # Create an (empty) layer for the hats

    hats = np.zeros(shape=(int(height), int(width), 4), dtype=np.uint8)

    # Add the hats to the hat's layer according the heads found

    for (x,y,w,h) in faces:
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        ratio = 3/4
        hat_height = int(ratio * h)
        hat_y = int(y - 3/4 * hat_height)
        addFixedImage(hats, hat1, x, hat_y, w, hat_height)

    ##############################################################

    # Mix the cover with the current camera frame

    # img = cv2.add(img, cover)

    img = cv2.addWeighted(img,
        0.9,
        cover,
        1,
        0)

    # Mix the hats layer with the current camera frame

    alpha = 1
    img = cv2.addWeighted(img,
        alpha,
        hats,
        1,
        0)

    # Show the constructed image (cam feed + cover)

    cv2.imshow('Xmas Postcard 2021', img)

    # Detect pressed keys

    key = cv2.waitKey(1)

    if key == 32: # [space] to save
        filename = saveImage(img)  
        print(f"{filename} saved!")
    if key == 27: # [esc] to quit
        break  

cv2.destroyAllWindows()