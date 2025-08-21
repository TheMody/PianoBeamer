import cv2
import numpy as np
import cv2.aruco as aruco
from config import *

# ─────────────────────────────────────────────────────────────────────────────
# 1)  Create an image that shows FOUR ArUco markers—one in every corner
# ─────────────────────────────────────────────────────────────────────────────
def create_four_marker_image(
        dict_name: int = aruco.DICT_4X4_50,
        marker_ids: tuple[int, int, int, int] = (0, 1, 2, 3),
        marker_px: int = marker_px,          # side length of each marker, in pixels
        border_px: int = border_px,           # white border around every marker
        canvas_size: tuple[int, int] = (b_height, b_width)  # (height, width)
) -> np.ndarray:
    """
    Returns a BGR image (uint8) containing four unique ArUco markers
    placed in the TL, TR, BR, BL corners, each surrounded by a white border.

    You can immediately write it to disk with cv2.imwrite(...) or show it
    with cv2.imshow(...).
    """
    # Build the dictionary
    aruco_dict = aruco.getPredefinedDictionary(dict_name)

    # Generate each marker once
    tiles = [aruco.generateImageMarker(aruco_dict, m_id, marker_px)
             for m_id in marker_ids]

    # Convert to 3-channel BGR and pad each tile with a white border
    tiles = [cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles]
    tiles = [cv2.copyMakeBorder(t, border_px, border_px,
                                border_px, border_px,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]

    # Prepare a white canvas
    H, W = canvas_size
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # Coordinates of the four corners (top-left, top-right, bottom-right, bottom-left)
    positions = [(0, 0),
                 (0, W - tiles[1].shape[1]),
                 (H - tiles[2].shape[0], W - tiles[2].shape[1]),
                 (H - tiles[3].shape[0], 0)]

    # Paste tiles onto the canvas
    for tile, (y, x) in zip(tiles, positions):
        h, w = tile.shape[:2]
        canvas[y:y + h, x:x + w] = tile

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 2)  Detect *those* four markers and return their (sub-pixel) centres
# ─────────────────────────────────────────────────────────────────────────────
def detect_four_markers(
        image,
        background_img: np.ndarray | None = None,
        dict_name: int = aruco.DICT_4X4_50,
        expected_ids: set[int] | None = None,
        refine_subpix: bool = True
):
    """
    Detects the four markers drawn by `create_four_marker_image`
    and returns a dict {marker_id: (cx, cy)} with sub-pixel centres.

    Raises ValueError if not all expected markers are found.
    """
    if expected_ids is None:
        expected_ids = {0, 1, 2, 3}

    aruco_dict = aruco.getPredefinedDictionary(dict_name)
    det_params = aruco.DetectorParameters()
    if refine_subpix:
        det_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(aruco_dict, det_params)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Gray Image", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if background_img is not None:
        # Subtract background image if provided
        gray = gray.astype(float)
        bg_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)#.astype(float)


        cv2.imshow("background Image", bg_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        gray = np.clip(gray - bg_gray, 0, 255) 
        #add some gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        gray = gray - np.min(gray)  # Ensure no negative values
        gray = (gray / np.max(gray) * 255).astype(np.uint8)  # Normalize to 0-255
        
    
    # cv2.imshow("Gray Image", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


   # gray = cv2.equalizeHist(gray)
    
    corners, ids, _ = detector.detectMarkers(gray.astype(np.uint8))

    if ids is None:
        raise ValueError("No ArUco markers detected at all.")

    found_ids = set(int(i) for i in ids.flatten())
    missing = expected_ids - found_ids
    if missing:
        raise ValueError(f"Missing marker IDs: {sorted(missing)}")

    # Compute each marker’s centre (〈cx, cy〉) at float precision
    centres = {}
    for c, m_id in zip(corners, ids.flatten()):
        pts = c.reshape(-1, 2)
        # Optional *extra* sub-pixel refinement
        if refine_subpix:
            pts = cv2.cornerSubPix(
                gray, pts,
                winSize=(5, 5), zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                          30, 1e-4)
            )
        #cx, cy = pts.mean(axis=0)

        if m_id == 0:
            cx,cy = pts[0]
        elif m_id == 1:
            cx,cy = pts[1]
        elif m_id == 2:
            cx,cy = pts[2]
        elif m_id == 3:
            cx,cy = pts[3]

        centres[int(m_id)] = (float(cx), float(cy))

    for i in range(len(centres)):
        centres[i] = np.asarray(centres[i])

    #TODO need to correct for boundary region. Region is needed for the detection algorithm
    centres[0][0] = centres[0][0] + border_px*(centres[0][0]-centres[1][0])/b_width
    centres[0][1] = centres[0][1] + border_px*(centres[0][1]-centres[3][1])/b_height

    centres[1][0] = centres[1][0] + border_px*(centres[1][0]-centres[0][0])/b_width
    centres[1][1] = centres[1][1] + border_px*(centres[1][1]-centres[2][1])/b_height

    centres[2][0] = centres[2][0] + border_px*(centres[2][0]-centres[3][0])/b_width
    centres[2][1] = centres[2][1] + border_px*(centres[2][1]-centres[1][1])/b_height

    centres[3][0] = centres[3][0] + border_px*(centres[3][0]-centres[2][0])/b_width
    centres[3][1] = centres[3][1] + border_px*(centres[3][1]-centres[0][1])/b_height

    return centres




if __name__ == "__main__":
    # Create an image with four markers
    img = create_four_marker_image()
    cv2.imwrite("images/four_markers.png", img)
    # cv2.imshow("Four Markers", img)
    # cv2.waitKey(0)

    # Detect the markers in the created image
    detected_centres = detect_four_markers(img)
    print("Detected marker centres:", detected_centres)

   # cv2.destroyAllWindows()