#!/usr/bin/env python3
"""
midi_keyboard_anim.py
=====================
Visualise a MIDI file on an animated piano keyboard.

• Current notes  → GREEN
• Next < window  → RED
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Set

import cv2
import mido
import numpy as np
import re
from config import *


class PianoKeyboardCV:
    """
    Draw an 88-key (or shorter) piano keyboard with OpenCV.
    Each key can be recoloured individually by MIDI number or note name.
    """
    WINNAME = 'Piano'

    # white-key MIDI remainders
    _WHITE_SET = (0, 2, 4, 5, 7, 9, 11)

    # semitone offsets within an octave
    _NOTE_BASES = {
        'C': 0,  'C#': 1, 'Db': 1,
        'D': 2,  'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5,  'F#': 6, 'Gb': 6,
        'G': 7,  'G#': 8, 'Ab': 8,
        'A': 9,  'A#': 10, 'Bb': 10,
        'B': 11,
    }
    _NOTE_RE = re.compile(r'([A-Ga-g](?:#|b)?)(-?\d+)\Z')

    def __init__(self, start_midi: int = 21, num_keys: int = 88):
        self.start = start_midi
        self.notes = list(range(start_midi, start_midi + num_keys))
        self.H = None
        # default colours (white / black)
        self._default_colours: Dict[int, Tuple[int, int, int]] = {
            n: (255, 255, 255) if self._is_white(n) else (0, 0, 0)
            for n in self.notes
        }
        self.colours = dict(self._default_colours)
        self.activated: Dict[int, float] = {n: 0.0 for n in self.notes}  # note → opacity (0.0 … 1.0)
        self._build_white_x()
        self._render()

    def reset_colours(self):
        """Revert all keys to their default white/black."""
        self.colours = dict(self._default_colours)

    def color_keys(self, mapping: Dict[int, Tuple[int, int, int]]):
        """
        mapping – {MIDI int | note str: colour}
        colour  – (B, G, R) tuple, '#RRGGBB', or basic colour name.
        """
        for key, colour in mapping.items():
            midi = self._to_midi(key)
            if midi in self.colours:
                self.colours[midi] = self._parse_colour(colour)
        self._render()

    def show(self, wait_ms: int = 1) -> bool:
        """Display and return *True* if user pressed Esc."""
        cv2.imshow(PianoKeyboardCV.WINNAME, self.img)
        key = cv2.waitKey(wait_ms) & 0xFF
        return key == 27                                 # Esc quits

    # internals ---------------------------------------------------------------
    def _build_white_x(self):
        self._white_x = {}
        x = 0
        for n in self.notes:
            if self._is_white(n):
                self._white_x[n] = x
                x += WHITE_W
        self.width = x
        self.height = WHITE_H

    def update_transform(self,H):
        self.H  = H

    def _render(self, add_frame = test):
        canvas = np.full((self.height, self.width, 3),
                         0, dtype=np.uint8)
        
        if add_frame:
            cpt1 = (0, 0)
            cpt2 = (self.width, 0)
            cpt3 = (self.width, self.height)
            cpt4 = (0, self.height)
            cv2.line(canvas, cpt1, cpt2, (255,255,255), 3, cv2.LINE_AA)
            cv2.line(canvas, cpt2, cpt3, (255,255,255), 3, cv2.LINE_AA)
            cv2.line(canvas, cpt3, cpt4, (255,255,255), 3, cv2.LINE_AA)
            cv2.line(canvas, cpt4, cpt1, (255,255,255), 3, cv2.LINE_AA)

        # pass 1 – white keys
        for n in self.notes:
            if self._is_white(n): #and self.activated[n] > 0.0:
                x = self._white_x[n]
                cv2.rectangle(canvas, (x, 0), (x + WHITE_W, int(WHITE_H)),
                              self._default_colours[n], thickness=-1)
                if self.activated[n] > 0.0:
                   cv2.rectangle(canvas, (x, 0), (x + WHITE_W, int(WHITE_H*self.activated[n])),
                              self.colours[n], thickness=-1)
                # cv2.rectangle(canvas, (x, 0), (x + WHITE_W, WHITE_H),
                #               (0, 0, 0), thickness=3)

        # pass 2 – black keys
        for n in self.notes:
            if not self._is_white(n):
                prev_white = n - 1
                while prev_white not in self._white_x:
                    prev_white -= 1
                x_left = (self._white_x[prev_white]
                          + WHITE_W - BLACK_W // 2)
                if self.colours[n] != self._default_colours[n]:
                    cv2.rectangle(canvas, (x_left, 0),
                                (x_left + BLACK_W, int(BLACK_H*self.activated[n])),
                                self.colours[n], thickness=-1)
                else:
                    cv2.rectangle(canvas, (x_left, 0),
                                (x_left + BLACK_W, BLACK_H),
                                self.colours[n], thickness=-1)
                # cv2.rectangle(canvas, (x_left, 0),
                #               (x_left + BLACK_W, BLACK_H),
                #               (0, 0, 0), thickness=3)

        if flip_keyboard:
            canvas = cv2.flip(canvas, -1)
                
        if self.H is not None:
            kb_new = np.zeros((b_height, b_width, 3), dtype=np.uint8)  # Create a blank image for the keyboard
            kb_new[:self.height, :self.width] = canvas
            display_img = cv2.warpPerspective(kb_new, self.H, (b_width, b_height))
            self.img = display_img
        else:
            self.img = canvas

    # utility ---------------------------------------------------------------
    @staticmethod
    def _is_white(midi: int) -> bool:
        return (midi % 12) in PianoKeyboardCV._WHITE_SET

    def _to_midi(self, key):
        if isinstance(key, int):
            return key
        m = self._NOTE_RE.fullmatch(key.strip())
        if not m:
            raise ValueError(f"Bad note name: {key!r}")
        base, octave = m.groups()
        base, octave = base.capitalize(), int(octave)
        if base not in self._NOTE_BASES:
            raise ValueError(f"Unknown pitch class: {base}")
        return self._NOTE_BASES[base] + (octave + 1) * 12

    @staticmethod
    def _parse_colour(c):
        if isinstance(c, (tuple, list)) and len(c) == 3:
            return tuple(int(v) for v in c[::-1])         # assume RGB → BGR
        if isinstance(c, str):
            c = c.strip()
            if c.startswith('#') and len(c) == 7:
                r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                return (b, g, r)
            simple = {
                'black': (0, 0, 0), 'white': (255, 255, 255),
                'red': (0, 0, 255), 'green': (0, 255, 0),
                'blue': (255, 0, 0), 'yellow': (0, 255, 255),
                'cyan': (255, 255, 0), 'magenta': (255, 0, 255),
                'grey': (128, 128, 128), 'gray': (128, 128, 128)
            }
            if c.lower() in simple:
                return simple[c.lower()]
        raise ValueError(f"Unsupported colour: {c!r}")


# ──────────────────────────────────────────────────────────────────────────────
#  MIDI → event list (seconds, note, on/off)
# ──────────────────────────────────────────────────────────────────────────────
NOTE_EVENT = Tuple[float, int, bool]     # (abs_time_sec, midi_note, is_on)


def tick_to_sec(ticks: int, tempo: int, ppq: int) -> float:
    """Ticks → seconds using the given tempo (μs/beat) and PPQ resolution."""
    return ticks * (tempo / 1_000_000) / ppq


def extract_events(mf: mido.MidiFile) -> List[NOTE_EVENT]:
    """Return a merged, tempo-aware list of (time, note, on/off) tuples."""
    track = mido.merge_tracks(mf.tracks)
    ppq = mf.ticks_per_beat
    tempo = 500_000                           # default 120 BPM
    abs_ticks = 0
    # for i in range(len(track)):
    #     if track[i].type == 'note_on':
    #         initial_time = tick_to_sec(track[0].time, tempo, ppq) makes problems due to time and rythm maybe not setting correctly
    initial_time = tick_to_sec(track[0].time, tempo, ppq)
    abs_time = LEADTIME_SONG - initial_time
    events: List[NOTE_EVENT] = []

    for msg in track:
        abs_ticks += msg.time
        abs_time += tick_to_sec(msg.time, tempo, ppq)

        if msg.type == 'set_tempo':
            tempo = msg.tempo
            continue

        if msg.type == 'note_on':
            if msg.velocity == 0:
                events.append((abs_time, msg.note, False))
            else:
                events.append((abs_time, msg.note, True))
        elif msg.type == 'note_off':
            events.append((abs_time, msg.note, False))

    return events


# ──────────────────────────────────────────────────────────────────────────────
#  Animation loop
# ──────────────────────────────────────────────────────────────────────────────
GREEN = (0, 255, 0)
RED   = (0,   0, 255)



def animate(events, keyboard, lookahead=0.5,playback_speed = 1, window_name='Piano'):
    """
    Realtime visualiser
        • GREEN – notes currently sounding
        • RED   – upcoming Note-On events (< lookahead s) with opacity proportional
                  to 1 – (Δt / lookahead).
    """

    try:
        t_start = time.time()
        idx, n_events = 0, len(events)
        active: set[int] = set()
        cv2.namedWindow(PianoKeyboardCV.WINNAME, cv2.WINDOW_NORMAL)          # create once, before first imshow
        cv2.setWindowProperty(PianoKeyboardCV.WINNAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)  
        while True:
            t_last = time.time()
            now = time.time() - t_start
            now *= playback_speed                      # speed up / slow down

            # ── 1. consume due events ────────────────────────────────────────────
            while idx < n_events and events[idx][0] <= now:
                _, note, is_on = events[idx]
                (active.add if is_on else active.discard)(note)
                idx += 1

            # ── 2. collect upcoming events & their Δt ────────────────────────────
            upcoming: dict[int, float] = {}           # note → seconds until start
            j = idx
            while j < n_events and (events[j][0] - now) <= lookahead:
                t_ev, note, is_on = events[j]
                if is_on and note not in active:
                    delta = t_ev - now
                    # keep the *soonest* Note-On if the note appears twice in window
                    upcoming[note] = min(delta, upcoming.get(note, lookahead))
                j += 1

            # ── 3. colour mapping ───────────────────────────────────────────────
            keyboard.reset_colours()

            # upcoming: blend default colour with red, opacity grows as Δt → 0
            for note, delta in upcoming.items():
                ratio = 1.0 - (delta / lookahead)        # 0 (far) … 1 (imminent)
                # default = keyboard._default_colours.get(
                #     note,
                #     (255, 255, 255) if keyboard._is_white(note) else (0, 0, 0)
                # )
                default = (0,0,0)
                red_pixel = (0, 0, 255)
                blended = tuple(
                    int((1 - ratio) * d + ratio * r) for d, r in zip(default, red_pixel)
                )
                keyboard.colours[note] = blended
                keyboard.activated[note] = ratio

            # active notes override with full-bright green
            for note in active:
                keyboard.colours[note] = GREEN

            # ── 4. render & handle UI ────────────────────────────────────────────
            keyboard._render()
            if keyboard.show():
                break                               # Esc pressed

            if idx >= n_events and not active:
                break                               # playback finished
            
        #  print("this iteration took", time.time() - t_last, "seconds")
            used_time = time.time() - t_last
            if used_time < 0.01:
                time.sleep(0.01 - used_time)                       # ~100 fps cap
    finally:
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():


    midi = mido.MidiFile("midi_files/Pirate.mid")
    events = extract_events(midi)

    kb = PianoKeyboardCV(start_midi=21, num_keys=NUM_KEYS)
    animate(events, kb, lookahead=LOOKAHEAD)


if __name__ == "__main__":
    main()
