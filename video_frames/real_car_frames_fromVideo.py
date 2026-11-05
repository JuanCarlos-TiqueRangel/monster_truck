import cv2
import numpy as np
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "wheelie_1.mp4"

# Exact REAL video frame used as the visible background.
# Change this if another frame has the background you prefer.
BACKGROUND_FRAME = 0

# Frames shown in the final wheelie sequence
SELECTED_FRAMES = [0, 7, 15, 22, 29, 36, 43, 52, 60, 68]

# 0.0 = invisible
# 1.0 = original vehicle appearance
FADE_LEVELS = [0.43, 0.49, 0.55, 0.62, 0.69, 0.76, 0.82, 0.88, 0.94, 1.00]


# ============================================================
# VEHICLE SEGMENTATION
# ============================================================

DIFF_THRESH = 30

LOWER_BLUE = np.array([95, 20, 15], dtype=np.uint8)
UPPER_BLUE = np.array([145, 255, 255], dtype=np.uint8)

MIN_COMPONENT_AREA = 120
STATIC_SCENE_SAMPLES = 60

CAR_PADDING_X = 140
CAR_PADDING_Y = 130

# Keep 0.0 if you want pixels outside the mask COMPLETELY unchanged.
FEATHER_SIGMA = 0.0


# ============================================================
# OUTPUT
# ============================================================

SHOW_PREVIEW = True
SHOW_DEBUG = False

CROP_RESULT = False
CROP_MARGIN_X = 100
CROP_MARGIN_Y = 80

# Save options
SAVE_PNG = False
SAVE_PDF = True

PNG_PATH = "wheelie_frames_background.png"
PDF_PATH = "wheelie_frames_background.pdf"

# Only affects physical size metadata in PDF.
# The image itself keeps its original pixel resolution.
PDF_DPI = 1200


# ============================================================
# READ ONE EXACT VIDEO FRAME
# ============================================================

def read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx}")

    return frame


# ============================================================
# BUILD MEDIAN SCENE
#
# IMPORTANT:
# This image is ONLY used for motion detection.
# It is NEVER used as the displayed background.
# ============================================================

def build_static_scene(video_path, n_samples=60):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_samples = min(n_samples, total_frames)

    sample_indices = np.linspace(
        0,
        total_frames - 1,
        n_samples,
        dtype=int
    )

    samples = []

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            samples.append(frame)

    cap.release()

    if not samples:
        raise RuntimeError("Could not obtain frames for static scene.")

    print(f"Building segmentation background from {len(samples)} frames...")

    return np.median(
        np.stack(samples, axis=0),
        axis=0
    ).astype(np.uint8)


# ============================================================
# BLUE VEHICLE PIXELS
# ============================================================

def get_blue_mask(frame, foreground):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # Must be blue AND moving
    blue_mask = cv2.bitwise_and(blue_mask, foreground)

    kernel3 = np.ones((3, 3), dtype=np.uint8)

    return cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel3,
        iterations=1
    )


# ============================================================
# KEEP BLUE COMPONENT BELONGING TO CAR
# ============================================================

def keep_car_blue_component(blue_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        blue_mask,
        connectivity=8
    )

    h, _ = blue_mask.shape

    best_label = -1
    best_area = -1

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < 15:
            continue

        _, cy = centroids[label_id]

        # Ignore blue/purple objects near ceiling/cameras
        if cy < 0.35 * h:
            continue

        if area > best_area:
            best_area = area
            best_label = label_id

    clean_blue = np.zeros_like(blue_mask)

    if best_label >= 0:
        clean_blue[labels == best_label] = 255

    return clean_blue


# ============================================================
# LOCAL REGION AROUND CAR
# ============================================================

def get_vehicle_roi_mask(blue_mask):
    ys, xs = np.where(blue_mask > 0)
    roi_mask = np.zeros_like(blue_mask)

    if len(xs) == 0:
        return roi_mask

    h, w = blue_mask.shape

    x1 = max(0, int(xs.min()) - CAR_PADDING_X)
    x2 = min(w, int(xs.max()) + CAR_PADDING_X + 1)

    y1 = max(0, int(ys.min()) - CAR_PADDING_Y)
    y2 = min(h, int(ys.max()) + CAR_PADDING_Y + 1)

    roi_mask[y1:y2, x1:x2] = 255

    return roi_mask


# ============================================================
# VEHICLE MASK
# ============================================================

def get_vehicle_mask(frame, static_scene):
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    kernel5 = np.ones((5, 5), dtype=np.uint8)
    kernel7 = np.ones((7, 7), dtype=np.uint8)
    kernel25 = np.ones((25, 25), dtype=np.uint8)

    # --------------------------------------------------------
    # MOTION
    # --------------------------------------------------------

    diff = cv2.absdiff(frame, static_scene)
    diff_score = np.max(diff, axis=2)

    foreground = (
        diff_score > DIFF_THRESH
    ).astype(np.uint8) * 255

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel3,
        iterations=1
    )

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
    # BLUE CAR SEED
    # --------------------------------------------------------

    blue_mask = get_blue_mask(frame, foreground)
    blue_mask = keep_car_blue_component(blue_mask)

    if cv2.countNonZero(blue_mask) == 0:
        return np.zeros_like(foreground)

    # --------------------------------------------------------
    # LOCAL CAR REGION
    # --------------------------------------------------------

    roi_mask = get_vehicle_roi_mask(blue_mask)
    foreground = cv2.bitwise_and(foreground, roi_mask)

    # Expand blue chassis toward tires/body
    blue_seed = cv2.dilate(
        blue_mask,
        kernel25,
        iterations=2
    )

    # --------------------------------------------------------
    # CONNECTED COMPONENTS
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8
    )

    vehicle_mask = np.zeros_like(foreground)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        component = (
            labels == label_id
        ).astype(np.uint8) * 255

        overlap = cv2.countNonZero(
            cv2.bitwise_and(component, blue_seed)
        )

        if overlap > 10:
            vehicle_mask = cv2.bitwise_or(
                vehicle_mask,
                component
            )

    vehicle_mask = cv2.morphologyEx(
        vehicle_mask,
        cv2.MORPH_CLOSE,
        kernel7,
        iterations=2
    )

    return vehicle_mask


# ============================================================
# REMOVE SMALL FRAGMENTS
# ============================================================

def remove_small_fragments(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean = np.zeros_like(mask)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        clean[labels == label_id] = 255

    return clean


# ============================================================
# MASK ALPHA
# ============================================================

def feather_mask(mask):
    alpha = mask.astype(np.float32) / 255.0

    if FEATHER_SIGMA > 0:
        alpha = cv2.GaussianBlur(
            alpha,
            (0, 0),
            FEATHER_SIGMA
        )

    return np.clip(alpha, 0.0, 1.0)


# ============================================================
# PASTE VEHICLE
#
# background_frame is the EXACT original camera frame.
# Pixels outside the vehicle mask remain untouched.
# ============================================================

def paste_vehicle_faded(canvas, frame, background_frame, mask, strength):
    if cv2.countNonZero(mask) == 0:
        return

    alpha = feather_mask(mask)[:, :, None]

    frame_f = frame.astype(np.float32)
    background_f = background_frame.astype(np.float32)
    canvas_f = canvas.astype(np.float32)

    # Fade only the vehicle toward the exact real background.
    vehicle = (
        strength * frame_f +
        (1.0 - strength) * background_f
    )

    result = (
        alpha * vehicle +
        (1.0 - alpha) * canvas_f
    )

    canvas[:] = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# DEBUG
# ============================================================

def show_debug(frame, mask, frame_idx):
    preview = frame.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        preview,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    cv2.imshow(f"Vehicle mask - frame {frame_idx}", preview)
    cv2.waitKey(0)
    cv2.destroyWindow(f"Vehicle mask - frame {frame_idx}")


# ============================================================
# CROP
# ============================================================

def crop_from_mask(image, sequence_mask):
    ys, xs = np.where(sequence_mask > 0)

    if len(xs) == 0:
        return image

    h, w = image.shape[:2]

    x1 = max(0, int(xs.min()) - CROP_MARGIN_X)
    x2 = min(w, int(xs.max()) + CROP_MARGIN_X + 1)

    y1 = max(0, int(ys.min()) - CROP_MARGIN_Y)
    y2 = min(h, int(ys.max()) + CROP_MARGIN_Y + 1)

    return image[y1:y2, x1:x2]


# ============================================================
# SAVE PNG
# ============================================================

def save_png(image, path):
    success = cv2.imwrite(
        path,
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )

    if not success:
        raise RuntimeError(f"Could not save PNG: {path}")

    print(f"Saved PNG: {path}")


# ============================================================
# SAVE PDF
# ============================================================

def save_pdf(image, path, dpi=300):
    # OpenCV uses BGR, PIL uses RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    pil_image.save(
        path,
        "PDF",
        resolution=dpi
    )

    print(f"Saved PDF: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    if len(SELECTED_FRAMES) != len(FADE_LEVELS):
        raise ValueError(
            "SELECTED_FRAMES and FADE_LEVELS must have the same length."
        )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    print()
    print(f"Video: {VIDEO_PATH}")
    print(f"Total frames: {total_frames}")
    print(f"Valid frame indices: 0 to {total_frames - 1}")
    print(f"FPS: {fps:.2f}")
    print(f"Background frame: {BACKGROUND_FRAME}")
    print(f"Selected frames: {SELECTED_FRAMES}")
    print()

    if BACKGROUND_FRAME < 0 or BACKGROUND_FRAME >= total_frames:
        cap.release()
        raise ValueError(
            f"BACKGROUND_FRAME must be between 0 and {total_frames - 1}."
        )

    for frame_idx in SELECTED_FRAMES:
        if frame_idx < 0 or frame_idx >= total_frames:
            cap.release()
            raise ValueError(
                f"Frame {frame_idx} does not exist. "
                f"Valid frames are 0 to {total_frames - 1}."
            )

    # ========================================================
    # MEDIAN IMAGE FOR SEGMENTATION ONLY
    # ========================================================

    static_scene = build_static_scene(
        VIDEO_PATH,
        STATIC_SCENE_SAMPLES
    )

    # ========================================================
    # EXACT ORIGINAL BACKGROUND
    # ========================================================

    background_frame = read_video_frame(
        VIDEO_PATH,
        BACKGROUND_FRAME
    )

    # The visible image begins as an untouched real frame.
    canvas = background_frame.copy()

    sequence_mask = np.zeros(
        background_frame.shape[:2],
        dtype=np.uint8
    )

    # ========================================================
    # VEHICLE POSES
    # ========================================================

    for frame_idx, strength in zip(
        SELECTED_FRAMES,
        FADE_LEVELS
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"WARNING: could not read frame {frame_idx}")
            continue

        mask = get_vehicle_mask(frame, static_scene)
        mask = remove_small_fragments(mask)

        n_pixels = cv2.countNonZero(mask)

        if n_pixels == 0:
            print(f"WARNING: no vehicle detected in frame {frame_idx}")
            continue

        sequence_mask = cv2.bitwise_or(
            sequence_mask,
            mask
        )

        paste_vehicle_faded(
            canvas,
            frame,
            background_frame,
            mask,
            strength
        )

        time_s = frame_idx / fps

        print(
            f"Frame {frame_idx:2d} | "
            f"t = {time_s:.3f} s | "
            f"strength = {strength:.2f} | "
            f"pixels = {n_pixels}"
        )

        if SHOW_DEBUG:
            show_debug(frame, mask, frame_idx)

    cap.release()

    # ========================================================
    # OPTIONAL CROP
    # ========================================================

    if CROP_RESULT:
        canvas = crop_from_mask(
            canvas,
            sequence_mask
        )

    # ========================================================
    # SAVE
    # ========================================================

    print()

    if SAVE_PNG:
        save_png(
            canvas,
            PNG_PATH
        )

    if SAVE_PDF:
        save_pdf(
            canvas,
            PDF_PATH,
            PDF_DPI
        )

    if not SAVE_PNG and not SAVE_PDF:
        print("WARNING: SAVE_PNG and SAVE_PDF are both False.")

    print(
        f"Output size: "
        f"{canvas.shape[1]} x {canvas.shape[0]}"
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    if SHOW_PREVIEW:
        cv2.imshow(
            "Wheelie sequence",
            canvas
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()