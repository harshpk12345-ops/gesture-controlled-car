import cv2
import mediapipe as mp
import requests
import threading
import time
import math
import numpy as np

CAMERA_IP = "192.168.1.50"
CAR_ESP_IP = "192.168.1.54"

STREAM_URL = f"http://{CAMERA_IP}:81/stream"
CAR_STREAM_URL = f"http://{CAR_ESP_IP}:81/stream"
COMMAND_URL = f"http://{CAR_ESP_IP}/command"
FLASH_ON_URL = f"http://{CAMERA_IP}/flashON"
FLASH_OFF_URL = f"http://{CAMERA_IP}/flashOFF"

HEARTBEAT_SECONDS = 0.5

# How many consecutive frames must agree on thumbs-up/thumbs-down before
# the flash actually toggles. Keeps a single misread frame from flipping
# the light. Note this is independent of GESTURE_STABILITY_FRAMES below --
# the flash and the drive command are two separate channels.
FLASH_STABILITY_FRAMES = 3

# Cosine-similarity margin (unitless, range -1..1) for thumb up/down
# detection, measured against the hand's own orientation axis rather than
# raw image y -- robust to hand tilt/rotation. 0.5 =~ thumb must point
# within ~60 degrees of the hand's up-axis to count.
THUMB_ALIGNMENT_MARGIN = 0.5

# Thumb must be extended at least this fraction of the hand's own length
# to be considered "pointing" anywhere -- filters out a curled thumb
# that's just resting near the base knuckle.
THUMB_MIN_EXTENSION_RATIO = 0.4

# How many consecutive frames must agree on a gesture before it's acted
# on. MediaPipe's per-frame finger detection can flicker for a frame or
# two (occlusion, motion blur), which without this would show up as the
# car briefly jerking into the wrong direction.
GESTURE_STABILITY_FRAMES = 3

# Margin (as a fraction of hand length) a fingertip must project past its
# PIP joint along the hand's own axis to count as "extended". Distance-
# independent, unlike a raw pixel threshold.
FINGER_EXTENSION_MARGIN_RATIO = 0.08

# How long a camera stream can go without a fresh successful read before
# we force a reconnect, even if cv2.VideoCapture still reports "open".
# Guards against a stalled MJPEG stream silently serving the same frame
# forever while showing as "CONNECTED".
STREAM_STALE_TIMEOUT = 3.0


class StreamGrabber:
    def __init__(self, url, stale_timeout=STREAM_STALE_TIMEOUT):
        self.url = url
        self.stale_timeout = stale_timeout
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.connected = self.cap.isOpened()
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.last_frame_time = time.time()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )
        self.thread.start()

    def _run(self):
        while self.running:

            if self.cap is None or not self.cap.isOpened():
                print("Reconnecting to ESP32-CAM...")

                if self.cap is not None:
                    self.cap.release()

                self.cap = cv2.VideoCapture(
                    self.url,
                    cv2.CAP_FFMPEG
                )

                # Keep the internal buffer at 1 frame so we always read the
                # freshest frame instead of draining a backlog -- reduces
                # perceived latency and stops stale frames piling up.
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self.connected = self.cap.isOpened()

                if not self.connected:
                    time.sleep(1)
                    continue

                self.last_frame_time = time.time()

            success, frame = self.cap.read()
            now = time.time()

            if success:
                with self.lock:
                    self.frame = frame
                    self.connected = True
                self.last_frame_time = now
            else:
                self.connected = False

                if self.cap is not None:
                    self.cap.release()

                self.cap = None
                time.sleep(0.5)
                continue

            # No successful read in stale_timeout seconds despite the
            # capture object still reporting "open" -- force a reconnect
            # rather than silently serving a frozen frame as "connected".
            if now - self.last_frame_time > self.stale_timeout:
                print("Stream stalled, forcing reconnect...")

                with self.lock:
                    self.connected = False

                if self.cap is not None:
                    self.cap.release()

                self.cap = None

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):
        self.running = False

        if self.thread.is_alive():
            self.thread.join(timeout=1)

        if self.cap is not None:
            self.cap.release()


print("Connecting to ESP32-CAM...")
print("Stream:", STREAM_URL)
print("Car ESP32:", CAR_ESP_IP)

grabber = StreamGrabber(STREAM_URL)
car_grabber = StreamGrabber(CAR_STREAM_URL)

print("Starting AI gesture control...")
print()
print("no fingers        = STOP")
print("index              = FORWARD")
print("index+middle       = BACKWARD")
print("index+middle+ring  = LEFT")
print("index+middle+ring+pinky = RIGHT")
print()
print("thumbs-up (fist)   = flash ON")
print("thumbs-down (fist) = flash OFF")
print()
print("Press Q or ESC to close.")


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# --- Hand-orientation-relative geometry helpers --------------------------
# Both finger-extension and thumb-direction detection are measured
# relative to the hand's OWN axis (wrist -> middle-finger MCP) rather than
# raw image coordinates. This keeps both robust to the hand being tilted
# or rotated relative to the camera, and to the hand's distance from the
# camera (margins are scaled by hand length, not fixed pixel/normalized
# amounts).

def get_hand_axis(landmarks):
    """Unit vector from wrist to middle-finger MCP -- the hand's own 'up'
    direction -- plus the hand's length, used to scale extension margins.
    Returns (None, 0.0) if the hand landmarks are degenerate."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    hand_len = math.hypot(dx, dy)

    if hand_len < 1e-6:
        return None, 0.0

    return (dx / hand_len, dy / hand_len), hand_len


FINGER_LANDMARKS = {
    "index": (8, 6),    # (tip, pip)
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}

# Only these exact combinations map to a command. Anything else (e.g.
# middle+ring with no index, or three-out-of-four fingers) falls through
# to STOP -- deliberate, since an ambiguous or incomplete gesture
# shouldn't be guessed at and used to drive the car.
FINGER_COMBO_TO_COMMAND = {
    frozenset(): "STOP",
    frozenset({"index"}): "FORWARD",
    frozenset({"index", "middle"}): "BACKWARD",
    frozenset({"index", "middle", "ring"}): "LEFT",
    frozenset({"index", "middle", "ring", "pinky"}): "RIGHT",
}


def _is_extended(landmarks, tip_idx, pip_idx, wrist, axis, hand_len):
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]

    tip_proj = (tip.x - wrist.x) * axis[0] + (tip.y - wrist.y) * axis[1]
    pip_proj = (pip.x - wrist.x) * axis[0] + (pip.y - wrist.y) * axis[1]

    return tip_proj > pip_proj + hand_len * FINGER_EXTENSION_MARGIN_RATIO


def get_extended_fingers(landmarks):
    """Set of finger names ('index', 'middle', 'ring', 'pinky') currently
    extended, measured relative to the hand's own orientation rather than
    raw image y -- robust to hand tilt/rotation."""
    axis, hand_len = get_hand_axis(landmarks)

    if axis is None:
        return set()

    wrist = landmarks[0]

    return {
        name
        for name, (tip_idx, pip_idx) in FINGER_LANDMARKS.items()
        if _is_extended(landmarks, tip_idx, pip_idx, wrist, axis, hand_len)
    }


def get_command_from_fingers(extended_fingers):
    return FINGER_COMBO_TO_COMMAND.get(frozenset(extended_fingers), "STOP")


STEERING_BY_COMMAND = {
    "FORWARD": ("CENTER", 90),
    "BACKWARD": ("CENTER", 90),
    "LEFT": ("LEFT", 45),
    "RIGHT": ("RIGHT", 135),
    "STOP": ("CENTER", 90),
}


def thumb_direction(landmarks):
    """
    "UP", "DOWN", or "NEUTRAL", based on the thumb's direction relative to
    the hand's OWN up-axis (wrist -> middle-finger MCP), rather than raw
    image y-coordinates. This keeps it reliable when the hand/fist is
    tilted or rotated relative to the camera, and normalizes by hand size
    so it doesn't drift with distance from the camera. Only meaningful
    when the other four fingers are folded (checked separately by the
    caller) -- a fist with the thumb sticking up or down.
    """
    axis, hand_len = get_hand_axis(landmarks)

    if axis is None:
        return "NEUTRAL"

    thumb_base = landmarks[2]
    thumb_tip = landmarks[4]

    thumb_dx = thumb_tip.x - thumb_base.x
    thumb_dy = thumb_tip.y - thumb_base.y
    thumb_len = math.hypot(thumb_dx, thumb_dy)

    # Thumb barely extended (curled into the fist) -- direction would be
    # noise, so don't report one.
    if thumb_len < hand_len * THUMB_MIN_EXTENSION_RATIO:
        return "NEUTRAL"

    thumb_dx /= thumb_len
    thumb_dy /= thumb_len

    # Dot product of the two unit vectors = cosine of the angle between
    # them. +1 means the thumb points the same way as the hand's own "up"
    # axis (thumbs-up), -1 means it points the opposite way
    # (thumbs-down), independent of how the fist is rotated on screen.
    alignment = thumb_dx * axis[0] + thumb_dy * axis[1]

    if alignment > THUMB_ALIGNMENT_MARGIN:
        return "UP"
    if alignment < -THUMB_ALIGNMENT_MARGIN:
        return "DOWN"
    return "NEUTRAL"


class FlashController:
    """
    Controls the gesture cam's own flash LED.

    Thumbs-up (fist + thumb pointing up), held for FLASH_STABILITY_FRAMES,
    turns the flash ON. It then STAYS on -- including if the hand leaves
    the frame entirely -- until an explicit thumbs-down (fist + thumb
    pointing down), also held for FLASH_STABILITY_FRAMES, turns it OFF.
    Losing the hand only resets the debounce counters, never the flash
    state itself.
    """

    def __init__(self, on_url, off_url, required_frames, timeout=0.5):
        self.on_url = on_url
        self.off_url = off_url
        self.required_frames = required_frames
        self.timeout = timeout
        self.state = False
        self._up_count = 0
        self._down_count = 0
        # Reused connection for flash requests, same rationale as the
        # command sender's Session -- avoids reconnecting on every toggle.
        self._session = requests.Session()

    def update(self, thumb_dir, fingers_folded):
        if not fingers_folded:
            # Any pose other than a fist doesn't count toward flash
            # debounce -- reset the counters but leave self.state alone.
            self._up_count = 0
            self._down_count = 0
            return self.state

        if thumb_dir == "UP":
            self._up_count += 1
            self._down_count = 0
        elif thumb_dir == "DOWN":
            self._down_count += 1
            self._up_count = 0
        else:
            self._up_count = 0
            self._down_count = 0

        if self._up_count >= self.required_frames and not self.state:
            self._send(True)
        elif self._down_count >= self.required_frames and self.state:
            self._send(False)

        return self.state

    def _send(self, want_on):
        self.state = want_on
        url = self.on_url if want_on else self.off_url

        def _worker():
            try:
                self._session.get(url, timeout=self.timeout)
            except requests.exceptions.RequestException:
                pass

        threading.Thread(target=_worker, daemon=True).start()


flash_controller = FlashController(FLASH_ON_URL, FLASH_OFF_URL, FLASH_STABILITY_FRAMES)


# --- Gesture stability filter -------------------------------------------
# Only switch the active command once the same reading has shown up for
# GESTURE_STABILITY_FRAMES frames in a row. A single flickered frame
# can't cause a spurious direction change.

class GestureStabilizer:
    def __init__(self, required_frames):
        self.required_frames = required_frames
        self.candidate = "STOP"
        self.candidate_count = 0
        self.stable_command = "STOP"

    def update(self, reading):
        if reading == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = reading
            self.candidate_count = 1

        if self.candidate_count >= self.required_frames:
            self.stable_command = self.candidate

        return self.stable_command


stabilizer = GestureStabilizer(GESTURE_STABILITY_FRAMES)


command_lock = threading.Lock()

current_command = "STOP"
stop_sender = False

# Shared status the main loop reads to show on the dashboard overlay.
# Always read AND written under command_lock -- see sender_loop().
car_connected = False
car_latency_ms = None


def sender_loop():

    global car_connected, car_latency_ms

    last_sent = None
    last_sent_time = 0

    # Reusing one Session keeps the TCP connection to the car alive
    # between requests instead of reconnecting every 100-500ms, which
    # cuts the latency of every single command send.
    session = requests.Session()

    while not stop_sender:

        with command_lock:
            command = current_command

        now = time.time()

        command_changed = command != last_sent

        heartbeat_due = (
            now - last_sent_time
        ) >= HEARTBEAT_SECONDS

        if command_changed or heartbeat_due:

            send_start = time.time()

            try:

                session.get(
                    COMMAND_URL,
                    params={"dir": command},
                    timeout=0.3
                )

                elapsed_ms = (time.time() - send_start) * 1000

                with command_lock:
                    car_connected = True
                    car_latency_ms = elapsed_ms

                if command_changed:
                    print("SENT:", command)

            except requests.exceptions.RequestException:
                with command_lock:
                    car_connected = False
                    car_latency_ms = None

            last_sent = command
            last_sent_time = now

        time.sleep(0.05)

    session.close()


sender_thread = threading.Thread(
    target=sender_loop,
    daemon=True
)

sender_thread.start()


# --- Virtual car (mirrors whatever command is actually being sent) -------

class VirtualCar:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.heading_deg = 90.0  # 90 = facing "up" on screen
        self.step = 3.0
        self.turn_rate = 4.0

    def update(self, command):
        if command == "FORWARD":
            self.x += self.step * math.cos(math.radians(self.heading_deg))
            self.y -= self.step * math.sin(math.radians(self.heading_deg))
        elif command == "BACKWARD":
            self.x -= self.step * math.cos(math.radians(self.heading_deg))
            self.y += self.step * math.sin(math.radians(self.heading_deg))
        elif command == "LEFT":
            self.heading_deg += self.turn_rate
            self.x += self.step * math.cos(math.radians(self.heading_deg))
            self.y -= self.step * math.sin(math.radians(self.heading_deg))
        elif command == "RIGHT":
            self.heading_deg -= self.turn_rate
            self.x += self.step * math.cos(math.radians(self.heading_deg))
            self.y -= self.step * math.sin(math.radians(self.heading_deg))
        # STOP -> no movement

    def clamp(self, width, height, margin=20):
        self.x = max(margin, min(width - margin, self.x))
        self.y = max(margin, min(height - margin, self.y))


def render_virtual_car(width, height, car):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(0, max(width, height), 40):
        cv2.line(canvas, (i, 0), (i, height), (40, 40, 40), 1)
        cv2.line(canvas, (0, i), (width, i), (40, 40, 40), 1)

    cx, cy = int(car.x), int(car.y)
    heading_rad = math.radians(car.heading_deg)

    length, half_width = 22, 7
    dx, dy = math.cos(heading_rad), -math.sin(heading_rad)
    px, py = -dy, dx

    nose = (int(cx + dx * length), int(cy + dy * length))
    left_rear = (
        int(cx - dx * length * 0.5 + px * half_width),
        int(cy - dy * length * 0.5 + py * half_width),
    )
    right_rear = (
        int(cx - dx * length * 0.5 - px * half_width),
        int(cy - dy * length * 0.5 - py * half_width),
    )

    pts = np.array([nose, left_rear, right_rear], dtype=np.int32)
    cv2.fillConvexPoly(canvas, pts, (60, 200, 255))
    cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1)

    cv2.rectangle(canvas, (0, 0), (width, 32), (0, 0, 0), -1)
    cv2.putText(canvas, "VIRTUAL CAR", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return canvas


def fit_frame(frame, target_w, target_h):
    """Resize a frame to fit inside (target_w, target_h) preserving aspect
    ratio, letterboxed on black so it can be safely hconcat'd with panels
    of a fixed height."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


virtual_car = VirtualCar(x=240, y=240)


try:

    while True:

        frame = grabber.get_frame()

        if frame is None:

            frame = np.zeros(
                (480, 640, 3),
                dtype=np.uint8
            )

            cv2.putText(
                frame,
                "CONNECTING TO ESP32-CAM...",
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            finger_count = 0
            extended_fingers = set()
            command = stabilizer.update("STOP")
            # No frame at all -- don't touch flash state, just let its
            # debounce counters lapse.
            flash_controller.update("NEUTRAL", False)

        else:

            frame = cv2.resize(
                frame,
                (640, 480)
            )

            frame = cv2.flip(
                frame,
                1
            )

            finger_count = 0
            extended_fingers = set()
            reading = "STOP"

            if grabber.connected:

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                result = hands.process(
                    rgb_frame
                )

                if result.multi_hand_landmarks:

                    hand = result.multi_hand_landmarks[0]

                    extended_fingers = get_extended_fingers(hand.landmark)
                    finger_count = len(extended_fingers)

                    reading = get_command_from_fingers(extended_fingers)

                    thumb_dir = thumb_direction(hand.landmark)
                    flash_controller.update(thumb_dir, finger_count == 0)

                    mp_draw.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )
                else:
                    # Hand not detected this frame -- decay flash debounce
                    # counters only, never change the flash state itself.
                    flash_controller.update("NEUTRAL", False)

            command = stabilizer.update(reading)

        with command_lock:
            current_command = command

        steering_dir, servo_angle = STEERING_BY_COMMAND[command]

        with command_lock:
            car_ok = car_connected
            latency = car_latency_ms

        cam_ok = grabber.connected
        flash_on = flash_controller.state

        cv2.rectangle(
            frame,
            (0, 0),
            (640, 150),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "AI GESTURE CONTROL",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "CAMERA: " + ("CONNECTED" if cam_ok else "DISCONNECTED"),
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 220, 80) if cam_ok else (60, 60, 230),
            1
        )

        cv2.putText(
            frame,
            "CAR ESP32: " + ("CONNECTED" if car_ok else "DISCONNECTED"),
            (250, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 220, 80) if car_ok else (60, 60, 230),
            1
        )

        lat_text = f"{latency:.0f} ms" if latency is not None else "--"
        cv2.putText(
            frame,
            "LATENCY: " + lat_text,
            (470, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1
        )

        fingers_text = "+".join(sorted(extended_fingers)) if extended_fingers else "NONE"
        cv2.putText(
            frame,
            "FINGERS: " + fingers_text,
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "COMMAND: " + command,
            (280, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "FLASH: " + ("ON" if flash_on else "OFF"),
            (470, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255) if flash_on else (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"STEERING: {steering_dir}   SERVO: {servo_angle} deg",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "index:FWD  +mid:BACK  +ring:LEFT  +pinky:RIGHT  none:STOP   |   thumbs-up: flash ON   thumbs-down: flash OFF",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1
        )

        # ---- Car camera feed ----
        car_frame = car_grabber.get_frame()

        if car_frame is None:
            car_view = np.zeros((480, 480, 3), dtype=np.uint8)
            cv2.putText(
                car_view,
                "NO CAR CAMERA",
                (110, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        else:
            car_view = fit_frame(car_frame, 480, 480)

        cv2.rectangle(car_view, (0, 0), (480, 32), (0, 0, 0), -1)
        cv2.putText(
            car_view,
            "CAR CAMERA",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ---- Virtual car (mirrors the command actually being sent) ----
        virtual_car.update(command)
        virtual_car.clamp(320, 480)
        vcar_view = render_virtual_car(320, 480, virtual_car)

        combined = cv2.hconcat([frame, car_view, vcar_view])

        cv2.imshow(
            "Gesture Controlled Car",
            combined
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            break

finally:

    with command_lock:
        current_command = "STOP"

    try:

        requests.get(
            COMMAND_URL,
            params={"dir": "STOP"},
            timeout=0.5
        )

    except requests.exceptions.RequestException:
        pass

    stop_sender = True

    sender_thread.join(
        timeout=1
    )

    grabber.stop()
    car_grabber.stop()

    hands.close()

    cv2.destroyAllWindows()

print("Program stopped.")
