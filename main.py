import cv2
import time
import os
from ultralytics import YOLO

# ================= LOG SETUP =================
BASE_DIR = os.getcwd()
LOG_DIR = os.path.join(BASE_DIR, "phone_logs")
os.makedirs(LOG_DIR, exist_ok=True)

RUN_TS = time.strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"phone_usage_{RUN_TS}.txt")

with open(LOG_FILE, "w") as f:
    f.write("PHONE USAGE LOG\n")
    f.write(f"Run started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 40 + "\n")

print(f"\n📁 Logging to: {LOG_FILE}\n")

# ================= MODELS ====================
yolo = YOLO("yolov8n.pt")

# ================= CONSTANTS =================
PHONE_CLASS = 67
PERSON_CLASS = 0

START_CONFIRM_SECONDS = 2.0   # phone must be visible this long to START
END_GRACE_SECONDS = 3.0       # phone must be gone this long to END
MIN_SESSION_SECONDS = 2.0     # ignore garbage blips

# ================= STATE =====================
using_phone = False
start_time = None

phone_visible_since = None
phone_last_seen = None

# ================= HELPERS ===================
def log_session(start, end):
    duration = end - start
    if duration < MIN_SESSION_SECONDS:
        return

    with open(LOG_FILE, "a") as f:
        f.write("\n" + "=" * 36 + "\n")
        f.write("PHONE USAGE SESSION\n")
        f.write(f"Start Time : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}\n")
        f.write(f"End Time   : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}\n")
        f.write(f"Duration   : {round(duration, 2)} seconds\n")
        f.write("=" * 36 + "\n")

    print(f"✅ Logged session: {round(duration,2)}s")

# ================= VIDEO =====================
cap = cv2.VideoCapture("myvideo.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    phone_detected = False
    person_detected = False
    phone_box = None

    results = yolo(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            if cls == PERSON_CLASS:
                person_detected = True
            elif cls == PHONE_CLASS:
                phone_detected = True
                phone_box = box.xyxy[0].cpu().numpy().astype(int)

    # ================= CORE LOGIC =================
    if phone_detected and person_detected:
        phone_last_seen = now

        if phone_visible_since is None:
            phone_visible_since = now

        # START SESSION
        if not using_phone and (now - phone_visible_since) >= START_CONFIRM_SECONDS:
            using_phone = True
            start_time = phone_visible_since
            print("▶ STARTED PHONE USAGE")

    else:
        phone_visible_since = None

        # END SESSION
        if using_phone and phone_last_seen and (now - phone_last_seen) >= END_GRACE_SECONDS:
            using_phone = False
            log_session(start_time, phone_last_seen)
            start_time = None
            phone_last_seen = None

    # ================= VISUAL =================
    if phone_box is not None:
        x1, y1, x2, y2 = phone_box
        color = (0, 255, 0) if using_phone else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame,
                    "USING PHONE" if using_phone else "PHONE",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    if using_phone:
        cv2.putText(frame,
                    f"⏱ {int(now - start_time)}s",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3)

    cv2.imshow("PHONE USAGE DETECTOR (STABLE)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= FINAL FLUSH =================
if using_phone and start_time and phone_last_seen:
    log_session(start_time, phone_last_seen)

with open(LOG_FILE, "a") as f:
    f.write("\nRun ended at " + time.strftime('%Y-%m-%d %H:%M:%S'))

cap.release()
cv2.destroyAllWindows()
