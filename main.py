import cv2
import numpy as np

def detect_keyboard(image):
    # detects a keyboard from an overhead view as best as possible
    # first tries to find the keyboard by finding the largest and brightest almost white area
   

    #apply convolution to the image which detects noise
    kernel = np.array([ [-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]], dtype=np.float32)
    noise_img = cv2.filter2D(image, -1, kernel)

    cv2.imshow('Keyboard Detection', noise_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Convert the image to grayscale
    #try to make a custom mask using numpy
    # Convert the image to numpy array

    image = np.array(image).astype(np.int32)


    print(image.shape)
    rg_diff = image[:, :, 2] - image[:, :, 1]
    rb_diff = image[:, :, 2] - image[:, :, 0]
    gb_diff = image[:, :, 1] - image[:, :, 0]
    max_diff = np.maximum(np.abs(rg_diff), np.maximum(np.abs(rb_diff), np.abs(gb_diff))).astype(np.uint8)

    
   #get color histogramm
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # Define a color range for white
    # lower_white = np.array([0, 0, 0])
    # upper_white = np.array([255, 50, 255])
    # mask = cv2.inRange(hsv_image, lower_white, upper_white)
    cv2.imshow('Keyboard Detection', max_diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Find contours in the mask
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     print("No contours found.")
    #     return None

    # cv2.drawContours(image, contours, -1, (0,255,0), 3)
    # Find the bounding square of the largest contour
   # x, y, w, h = cv2.boundingRect(largest_contour)

    # display the image and the bounding box
    # print("Bounding box coordinates:", x, y, w, h)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Keyboard Detection', image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return largest_contour

if __name__ == "__main__":
    # Load an image
    image = cv2.imread('example.jpg')
    # rescale image to a reasonable size
    if image is None:
        print("Error: Could not load image.")
    else:
        height, width = image.shape[:2]
        scale_factor = 800 / max(height, width)
        image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
        # Detect keyboard
        keyboard_contour = detect_keyboard(image)
        if keyboard_contour is not None:
            print("Keyboard detected.")
        else:
            print("No keyboard detected.")

    
    