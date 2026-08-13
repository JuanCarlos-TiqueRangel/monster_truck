import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "obstacles.mp4"
OUT_PATH = "obstacles_frames_fade.png"

# Frames used for the sequence
# SELECTED_FRAMES = [55, 60, 70, 105, 140, 175, 210, 220]

SELECTED_FRAMES = [55, 70, 85, 105, 120, 145, 170, 210]

# 0.0 = completely white
# 1.0 = original vehicle color
FADE_LEVELS = [0.01, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84, 1.00]

# ------------------------------------------------------------
# Vehicle segmentation
# ------------------------------------------------------------

DIFF_THRESH = 14

LOWER_BLUE = np.array([105, 50, 15], dtype=np.uint8)
UPPER_BLUE = np.array([135, 255, 255], dtype=np.uint8)

MIN_COMPONENT_AREA = 150

# Number of frames used to estimate the static scene
STATIC_SCENE_SAMPLES = 50

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

SHOW_PREVIEW = True

CROP_RESULT = True
CROP_MARGIN_X = 80
CROP_MARGIN_Y = 60


# ============================================================
# BUILD CLEAN MUJOCO BACKGROUND
# ============================================================

def build_clean_background(frame):
    h, w = frame.shape[:2]

    border_width = max(20, int(0.10 * w))

    left = frame[:, :border_width]
    right = frame[:, -border_width:]

    border_samples = np.concatenate(
        (left, right),
        axis=1
    )

    # Estimate background color independently for each row
    row_colors = np.median(
        border_samples,
        axis=1
    ).astype(np.uint8)

    background = np.repeat(
        row_colors[:, None, :],
        w,
        axis=1
    )

    return background


# ============================================================
# BUILD STATIC SCENE FROM THE VIDEO
# ============================================================

def build_static_scene(video_path, n_samples=50):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    sample_indices = np.linspace(
        0,
        total_frames - 1,
        n_samples,
        dtype=int
    )

    samples = []

    for frame_idx in sample_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        ret, frame = cap.read()

        if ret:
            samples.append(frame)

    cap.release()

    if len(samples) == 0:
        raise RuntimeError(
            "Could not obtain frames for static scene."
        )

    print(
        f"Building static scene from "
        f"{len(samples)} frames..."
    )

    # Median removes the moving vehicle while preserving
    # static objects such as the obstacle.
    static_scene = np.median(
        np.stack(samples, axis=0),
        axis=0
    ).astype(np.uint8)

    return static_scene


# ============================================================
# STATIC OBSTACLE MASK
# ============================================================

def get_static_object_mask(static_scene):

    clean_background = build_clean_background(
        static_scene
    )

    diff = cv2.absdiff(
        static_scene,
        clean_background
    )

    diff_score = np.max(
        diff,
        axis=2
    )

    mask = (
        diff_score > DIFF_THRESH
    ).astype(np.uint8) * 255

    kernel3 = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    kernel5 = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    # Remove isolated noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel3,
        iterations=1
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel5,
        iterations=2
    )

    # --------------------------------------------------------
    # Remove very small or extremely wide components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    clean_mask = np.zeros_like(mask)

    for label_id in range(1, num_labels):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        width = stats[
            label_id,
            cv2.CC_STAT_WIDTH
        ]

        height = stats[
            label_id,
            cv2.CC_STAT_HEIGHT
        ]

        if area < MIN_COMPONENT_AREA:
            continue

        # Ignore long thin MuJoCo lines
        if height > 0 and width / height > 6.0:
            continue

        clean_mask[
            labels == label_id
        ] = 255

    return clean_mask


# ============================================================
# VEHICLE MASK
# ============================================================

def get_vehicle_mask(frame, static_scene):

    kernel3 = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    kernel5 = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    kernel7 = np.ones(
        (7, 7),
        dtype=np.uint8
    )

    kernel15 = np.ones(
        (15, 15),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # MOVING OBJECTS
    #
    # Since static_scene already contains the obstacle,
    # subtracting it leaves mainly the moving car.
    # --------------------------------------------------------

    diff = cv2.absdiff(
        frame,
        static_scene
    )

    diff_score = np.max(
        diff,
        axis=2
    )

    foreground = (
        diff_score > DIFF_THRESH
    ).astype(np.uint8) * 255

    # Remove isolated pixels
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel3,
        iterations=1
    )

    # Fill holes in vehicle
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel5,
        iterations=2
    )

    foreground = cv2.dilate(
        foreground,
        kernel3,
        iterations=1
    )

    # --------------------------------------------------------
    # BLUE VEHICLE SEED
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    blue_mask = cv2.inRange(
        hsv,
        LOWER_BLUE,
        UPPER_BLUE
    )

    # VERY IMPORTANT:
    #
    # Blue must also be moving.
    #
    # Therefore the static blue obstacle cannot become
    # a seed for the vehicle detector.
    blue_mask = cv2.bitwise_and(
        blue_mask,
        foreground
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel3,
        iterations=1
    )

    # Expand blue vehicle parts so they intersect with
    # the full chassis/wheels component.
    blue_seed = cv2.dilate(
        blue_mask,
        kernel15,
        iterations=2
    )

    # --------------------------------------------------------
    # CONNECTED COMPONENTS
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8
        )
    )

    vehicle_mask = np.zeros_like(
        foreground
    )

    for label_id in range(
        1,
        num_labels
    ):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area < MIN_COMPONENT_AREA:
            continue

        component = (
            labels == label_id
        ).astype(np.uint8) * 255

        overlap = cv2.countNonZero(
            cv2.bitwise_and(
                component,
                blue_seed
            )
        )

        # Only components connected to the moving
        # blue parts of the vehicle are accepted.
        if overlap > 10:

            vehicle_mask = cv2.bitwise_or(
                vehicle_mask,
                component
            )

    # Connect wheels, chassis, cage, etc.
    vehicle_mask = cv2.morphologyEx(
        vehicle_mask,
        cv2.MORPH_CLOSE,
        kernel7,
        iterations=2
    )

    return vehicle_mask


# ============================================================
# REMOVE SHADOW FRAGMENTS
# ============================================================

def remove_shadow_fragments(mask):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    clean = np.zeros_like(mask)

    for label_id in range(
        1,
        num_labels
    ):

        width = stats[
            label_id,
            cv2.CC_STAT_WIDTH
        ]

        height = stats[
            label_id,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area < 100:
            continue

        # Remove extremely wide/thin shadow pieces.
        if height > 0 and width / height > 5.0:
            continue

        clean[
            labels == label_id
        ] = 255

    return clean


# ============================================================
# PASTE STATIC OBJECT
# ============================================================

def paste_static_object(
    canvas,
    static_scene,
    mask
):

    pixels = mask > 0

    canvas[pixels] = static_scene[pixels]


# ============================================================
# PASTE FADED VEHICLE
# ============================================================

def paste_vehicle_faded(
    canvas,
    frame,
    mask,
    strength
):

    pixels = mask > 0

    if not np.any(pixels):
        return

    vehicle = frame[
        pixels
    ].astype(np.float32)

    white = np.full_like(
        vehicle,
        255.0
    )

    # Fade vehicle toward white
    faded_vehicle = (
        strength * vehicle
        +
        (1.0 - strength) * white
    )

    # Later poses overwrite earlier poses
    canvas[pixels] = np.clip(
        faded_vehicle,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# CROP WHITE SPACE
# ============================================================

def crop_white_space(image):

    non_white = np.any(
        image < 248,
        axis=2
    )

    ys, xs = np.where(
        non_white
    )

    if len(xs) == 0:
        return image

    h, w = image.shape[:2]

    x1 = max(
        0,
        int(xs.min()) - CROP_MARGIN_X
    )

    x2 = min(
        w,
        int(xs.max()) + CROP_MARGIN_X + 1
    )

    y1 = max(
        0,
        int(ys.min()) - CROP_MARGIN_Y
    )

    y2 = min(
        h,
        int(ys.max()) + CROP_MARGIN_Y + 1
    )

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    if len(SELECTED_FRAMES) != len(FADE_LEVELS):
        raise ValueError(
            "SELECTED_FRAMES and FADE_LEVELS "
            "must have the same length."
        )

    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    print()
    print(f"Video: {VIDEO_PATH}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Selected frames: {SELECTED_FRAMES}")
    print(f"Fade levels: {FADE_LEVELS}")
    print()

    # Check selected frames
    for frame_idx in SELECTED_FRAMES:

        if frame_idx >= total_frames:
            cap.release()

            raise ValueError(
                f"Frame {frame_idx} does not exist. "
                f"Video only has {total_frames} frames."
            )

    # --------------------------------------------------------
    # BUILD STATIC SCENE
    # --------------------------------------------------------

    static_scene = build_static_scene(
        VIDEO_PATH,
        STATIC_SCENE_SAMPLES
    )

    h, w = static_scene.shape[:2]

    # --------------------------------------------------------
    # CREATE WHITE CANVAS
    # --------------------------------------------------------

    canvas = np.full(
        (h, w, 3),
        255,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # DETECT STATIC OBSTACLE
    # --------------------------------------------------------

    static_mask = get_static_object_mask(
        static_scene
    )

    # Put static obstacle on figure exactly once
    paste_static_object(
        canvas,
        static_scene,
        static_mask
    )

    print(
        "Static pixels:",
        cv2.countNonZero(static_mask)
    )

    # --------------------------------------------------------
    # FRAME -> FADE LEVEL
    # --------------------------------------------------------

    fade_by_frame = dict(
        zip(
            SELECTED_FRAMES,
            FADE_LEVELS
        )
    )

    # --------------------------------------------------------
    # PROCESS SELECTED FRAMES
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx not in fade_by_frame:
            frame_idx += 1
            continue

        strength = fade_by_frame[
            frame_idx
        ]

        # Detect moving car
        mask = get_vehicle_mask(
            frame,
            static_scene
        )

        # Remove isolated shadow pieces
        mask = remove_shadow_fragments(
            mask
        )

        n_pixels = cv2.countNonZero(
            mask
        )

        if n_pixels == 0:

            print(
                f"WARNING: no vehicle detected "
                f"in frame {frame_idx}"
            )

        else:

            paste_vehicle_faded(
                canvas,
                frame,
                mask,
                strength
            )

            time_s = frame_idx / fps

            print(
                f"Frame {frame_idx:3d} | "
                f"t = {time_s:.3f} s | "
                f"strength = {strength:.2f} | "
                f"pixels = {n_pixels}"
            )

        frame_idx += 1

    cap.release()

    # --------------------------------------------------------
    # CROP
    # --------------------------------------------------------

    if CROP_RESULT:

        canvas = crop_white_space(
            canvas
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    success = cv2.imwrite(
        OUT_PATH,
        canvas,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3
        ]
    )

    if not success:
        raise RuntimeError(
            f"Could not save image: {OUT_PATH}"
        )

    print()
    print(f"Saved: {OUT_PATH}")
    print(
        f"Output size: "
        f"{canvas.shape[1]} x {canvas.shape[0]}"
    )

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    if SHOW_PREVIEW:

        cv2.imshow(
            "Obstacle sequence - temporal fade",
            canvas
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()