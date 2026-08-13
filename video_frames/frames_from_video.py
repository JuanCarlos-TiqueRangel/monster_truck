import cv2
import numpy as np
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "flip.mov"
OUT_PATH = "flip_sequence.pdf"

KEYFRAME_INDICES = [0, 20, 40, 48, 59]

BACKGROUND_THRESHOLD = 240

MARGIN_X = 25
MARGIN_Y = 20

GAP = 50


# ============================================================
# LOAD VIDEO
# ============================================================

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    print(f"Loaded {len(frames)} frames.")

    return frames


# ============================================================
# EXTRACT VEHICLE + SHADOW
# ============================================================

def extract_vehicle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Keep dark vehicle/shadow pixels
    mask = (gray < BACKGROUND_THRESHOLD).astype(np.uint8) * 255

    h, w = mask.shape

    # --------------------------------------------------------
    # REMOVE VERTICAL VIDEO/FRAME LINES
    # --------------------------------------------------------

    # Number of foreground pixels in every column
    column_pixels = np.sum(mask > 0, axis=0)

    # A vertical line spanning almost the whole image
    bad_columns = column_pixels > 0.90 * h

    # Remove those columns
    mask[:, bad_columns] = 0

    # Also ignore a tiny border around the original frame
    BORDER = 5
    mask[:, :BORDER] = 0
    mask[:, -BORDER:] = 0
    mask[:BORDER, :] = 0
    mask[-BORDER:, :] = 0

    # Clean noise
    kernel = np.ones((3, 3), dtype=np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # VEHICLE BOUNDING BOX
    # --------------------------------------------------------

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise RuntimeError("Could not detect vehicle.")

    x1 = max(0, int(xs.min()) - MARGIN_X)
    x2 = min(w, int(xs.max()) + MARGIN_X + 1)

    y1 = max(0, int(ys.min()) - MARGIN_Y)
    y2 = min(h, int(ys.max()) + MARGIN_Y + 1)

    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()

    return crop, crop_mask


# ============================================================
# CREATE ONE SINGLE WHITE CANVAS
# ============================================================

def create_sequence(frames):
    vehicles = []

    for idx in KEYFRAME_INDICES:
        if idx >= len(frames):
            raise ValueError(f"Frame {idx} does not exist.")

        crop, mask = extract_vehicle(frames[idx])
        vehicles.append((crop, mask))

    max_height = max(crop.shape[0] for crop, _ in vehicles)
    widths = [crop.shape[1] for crop, _ in vehicles]
    total_width = sum(widths) + GAP * (len(vehicles) - 1)

    # ONE continuous white image
    output = np.full((max_height, total_width, 3), 255, dtype=np.uint8)
    x_offset = 0

    for crop, mask in vehicles:
        h, w = crop.shape[:2]
        y_offset = (max_height - h) // 2
        region = output[y_offset:y_offset + h, x_offset:x_offset + w]
        pixels = mask > 0

        # Copy ONLY detected vehicle/shadow pixels
        region[pixels] = crop[pixels]
        x_offset += w + GAP

    return output


# ============================================================
# MAIN
# ============================================================

def main():
    frames = load_video(VIDEO_PATH)

    output = create_sequence(frames)

    # OpenCV uses BGR, Pillow uses RGB
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    pdf_image = Image.fromarray(output_rgb)
    pdf_image.save("flip_sequence.pdf", "PDF", resolution=1000.0)

    print(f"Saved: {OUT_PATH}")
    print(f"Output size: {output.shape[1]} x {output.shape[0]}")

    cv2.imshow("Flip sequence", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()