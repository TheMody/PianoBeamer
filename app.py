from json import load
from pathlib import Path
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, session, send_file
)
from markupsafe import Markup  
from main import setup_and_calibrate, play_song, recalibrate
from collections import deque
import builtins, functools
import numpy as np
import os
import cv2
import io
from config import parameter_file, NUM_KEYS
from utils import save_parameters, load_parameters
from keyboard_vis_cv import PianoKeyboardCV
from download_midi import download_first_midi
from camera_undistort import undistort_camera

# ----------------------------------
# CONFIG ---------------------------------------------------------------
# ----------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent.resolve()
FILES_DIR     = BASE_DIR / "midi_files"   # folder whose contents you list
FILES_DIR.mkdir(exist_ok=True)               # create on first run if absent
LOG_BUFFER = deque(maxlen=500)

app           = Flask(__name__)
app.secret_key = "replace-me"                # required for flash messages

kb = None

# ------------------------------------------------------------------
# 2. Wrap built-in print so every print() goes into the buffer
# ------------------------------------------------------------------
_builtin_print = builtins.print

def tee_print(*args, **kwargs):
    _builtin_print(*args, **kwargs)                   # console as usual
    message = " ".join(map(str, args))
    LOG_BUFFER.append(message)

builtins.print = tee_print   
latest_snapshot = None 

if os.path.exists(parameter_file):
    print(f"Loading parameters from {parameter_file}")
    keyboard_contour, beamer_contour, camera_distortion  = load_parameters()  # load initial parameters if available
    edge_sets = {
        "keyboard": keyboard_contour,
        "beamer":   beamer_contour,
    }
    kb = PianoKeyboardCV(start_midi=21, num_keys=NUM_KEYS)
    recalibrate(kb, keyboard_contour, beamer_contour)
else:
    edge_sets = {
        "keyboard":  [(10, 10), (630, 10), (630, 470), (10, 470)],   # default
        "beamer":    [(20, 20), (620, 30), (620, 460), (30, 460)],
    }

# --- helper to JSON-serialize tuples --------------------------------
def _as_dict():
    return {k: [dict(x=int(x), y=int(y)) for x, y in pts] for k, pts in edge_sets.items()}

# ----------------------------------
# DOMAIN-LEVEL “EVENT” FUNCTIONS --------------------------------------
# ----------------------------------------------------------------------
def event_one():
    """Placeholder for first button."""
    global kb, latest_snapshot, edge_sets
    kb,latest_snapshot, keyboard_edges, beamer_edges  = setup_and_calibrate()
    edge_sets['keyboard'] = keyboard_edges
    edge_sets['beamer'] = beamer_edges
    save_parameters(keyboard_edges, beamer_edges)  # save the updated parameters
  #  print("Camera and beamer setup calibrated successfully.")
    pass

def event_two():
    """Placeholder for second button."""
    undistort_camera()
    pass

def process_file(file_path: Path, speed: float = 1.0):
    """Placeholder for file-specific action."""
    global kb
    if kb is None:             # guard in case user skipped “Setup”
        flash("Please run setup & calibration first.", "error")
        return
    play_song(file_path, kb, playback_speed=speed)
    pass



# ----------------------------------
# ROUTES ---------------------------------------------------------------
# ----------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    speed = session.get("speed", 1.0)
    files = sorted(p.name for p in FILES_DIR.iterdir() if p.is_file())
    return render_template(
        "index.html",
        files=files,
        FILES_DIR=FILES_DIR ,
        speed=speed           
    )

@app.route("/logs")
def get_logs():
    return "\n".join(LOG_BUFFER), 200, {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache",
    }


@app.route("/process_file", methods=["POST"])
def process_selected():
    filename = request.form.get("selected_file")
    speed    = float(request.form.get("speed", session.get("speed", 1.0)))
    session["speed"] = speed 

    if not filename:
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    file_path = FILES_DIR / filename
    if not file_path.exists():
        flash("File disappeared; reload.", "error")
    else:
        process_file(file_path, speed)                 # <- pass speed
        flash(Markup(f"Playing <code>{filename}</code> at {speed:.1f}×"), "success")
    return redirect(url_for("index"))

@app.route("/trigger/<name>", methods=["POST"])
def trigger(name):
    match name:
        case "event_one":
            event_one()
            flash("Calibrated camera and beamer setup.", "success")
        case "event_two":
            event_two()
            flash("Undistort Successfull", "success")
        case _:
            flash(f"Unknown event {name!r}", "error")
    return redirect(url_for("index"))

# ────────────────────────────────────────────────
# Route that receives the form submission
# ────────────────────────────────────────────────
@app.route("/download_url", methods=["POST"])
def download_url():
    url = request.form.get("url")
    if not url:
        flash("No URL provided.", "error")
        return redirect(url_for("index"))
    try:
        download_first_midi(url)               
        flash(f"URL processed: {url}", "success")
    except Exception as e:
        flash(f"Error processing URL: {e}", "error")
    return redirect(url_for("index"))

@app.get("/edge_points")
def get_edge_points():
    return _as_dict()     # Flask will jsonify the dict automatically

@app.post("/edge_points")
def update_edge_points():
    global kb, edge_sets
    data = request.json
    edge_sets['keyboard'] = [(p['x'], p['y']) for p in data['keyboard']]
    edge_sets['beamer']   = [(p['x'], p['y']) for p in data['beamer']]

    recalibrate(kb, edge_sets['keyboard'], edge_sets['beamer'])
   # flash("Edge points updated ✔", "success")
    return {"ok": True}

@app.route("/snapshot")
def snapshot():
    """
    Return the most recent snapshot as PNG.
    Called once per page load by refreshSnapshot() in index.html.
    """
    global latest_snapshot

    if latest_snapshot is None:
        # Fallback: return a 1×1 transparent PNG to avoid browser errors
        latest_snapshot = np.zeros((1, 1, 4), dtype=np.uint8)

    # Encode the NumPy image → PNG bytes in memory
    ok, buf = cv2.imencode(".png", latest_snapshot)
    if not ok:
        # If encode failed, fall back to empty image
        buf = cv2.imencode(".png", np.zeros((1, 1, 4), dtype=np.uint8))[1]

    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/png",
        max_age=0,                # disable caching
        as_attachment=False
    )


# ----------------------------------
# ENTRY POINT ----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded = False)
