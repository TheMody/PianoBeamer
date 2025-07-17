#beamer parameters
b_height = 1080
b_width = 1920

#camera parameters
c_height = 1080
c_width = 1920

#marker parameters
marker_px = 240  # side length of each marker, in pixels
border_px = 50  # white border around every marker
marker_Mode = "white" # "marker" white
marker_threshold = 50 #set threshold higher for low light conditions, lower for bright conditions

#keyboard parameters
NUM_KEYS = 85  # number of keys on the keyboard
WHITE_W, WHITE_H = 20, 100          # white-key geometry  (pixels)
BLACK_W, BLACK_H = 12, 65           # black-key geometry  (pixels)
LOOKAHEAD = 2.0  # seconds to look ahead in the animation
flip_keyboard = False # flip the keyboard horizontally
BLACK_KEYS_COLOR = (0,255,255)
WHITE_KEYS_COLOR = (0,0,255)

#other parameters
LEADTIME_SONG = 2.0
WEBCAM_ID = 0
test = True