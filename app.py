from pathlib import Path
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, session
)
from markupsafe import Markup  
from main import setup_and_calibrate, play_song
from collections import deque
import builtins, functools
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

# ----------------------------------
# DOMAIN-LEVEL “EVENT” FUNCTIONS --------------------------------------
# (leave bodies empty for now – insert real logic later)
# ----------------------------------------------------------------------
def event_one():
    """Placeholder for first button."""
    global kb
    kb = setup_and_calibrate()
    pass

def event_two():
    """Placeholder for second button."""
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
            flash("Event 1 executed.", "success")
        case "event_two":
            event_two()
            flash("Event 2 executed.", "success")
        case _:
            flash(f"Unknown event {name!r}", "error")
    return redirect(url_for("index"))



# ----------------------------------
# ENTRY POINT ----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Use debug=True only during development
    app.run(debug=False, host="0.0.0.0", port=5000, threaded = False)
