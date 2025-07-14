#!/usr/bin/env python3
"""
midi_print.py – Print note events (names, channel, velocity, time)
===============================================================

Requires:  pip install mido python-rtmidi   # rtmidi is only for realtime; not used here
"""
import argparse
from pathlib import Path
from typing import Optional, List

import mido


# ---------- Helpers -----------------------------------------------------------
NOTE_NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F',
              'F♯', 'G', 'G♯', 'A', 'A♯', 'B']


def note_name(number: int) -> str:
    """Return e.g. 60 -> C4"""
    octave, idx = divmod(number, 12)
    return f'{NOTE_NAMES[idx]}{octave - 1}'     # MIDI octave 0 starts at C-1


def tick2sec(ticks: int, tempo: int, ticks_per_beat: int) -> float:
    """
    Convert *ticks* to seconds using the current *tempo*
    (μs per beat) and the file's *ticks_per_beat* resolution.
    """
    return ticks * (tempo / 1_000_000) / ticks_per_beat


# ---------- Core printing logic ----------------------------------------------
def print_midi(
    mf: mido.MidiFile,
    track_no: Optional[int],
    mode: str,
    units: str,
) -> None:
    """
    Iterate over chosen tracks (or all-merged) and print note events.
    mode  : 'absolute' | 'relative'
    units : 'seconds'  | 'ticks'
    """
    tracks: List[mido.MidiTrack]
    if track_no is None:
        # merge all tracks so that delta-times make sense globally
        tracks = [mido.merge_tracks(mf.tracks)]
    else:
        tracks = [mf.tracks[track_no]]

    ticks_per_beat = mf.ticks_per_beat
    default_tempo = 500_000           # μs per quarter note, per MIDI spec
    current_tempo = default_tempo

    abs_time_ticks = 0        # running total in ticks
    abs_time_secs = 0.0       # running total in seconds

    for tr in tracks:
        for msg in tr:
            abs_time_ticks += msg.time

            # update tempo if necessary (affects *future* events)
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo

            if units == 'seconds':
                abs_time_secs += tick2sec(msg.time, current_tempo,
                                          ticks_per_beat)

            if msg.type in ('note_on', 'note_off'):
                # skip “note_on velocity 0” duplicates if you like, but we show both types
                when = (
                    f'{abs_time_secs:9.3f}s'
                    if units == 'seconds'
                    else f'{abs_time_ticks:6d} ticks'
                )
                if mode == 'relative':
                    when = (
                        f'{tick2sec(msg.time, current_tempo, ticks_per_beat):9.3f}s'
                        if units == 'seconds'
                        else f'{msg.time:6d} ticks'
                    )

                kind = 'ON ' if msg.type == 'note_on' and msg.velocity > 0 else 'OFF'
                print(f'{when}  ch{msg.channel + 1:02}  {kind}  '
                      f'{note_name(msg.note):>3}  vel={msg.velocity:3d}')


# ---------- CLI --------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print MIDI NoteOn/Off events with timing."
    )
    ap.add_argument("midi_file", type=Path,
                    help="Path to a .mid file")
    ap.add_argument("-m", "--mode", choices=["absolute", "relative"],
                    default="absolute",
                    help="Show cumulative (absolute) time or delta (relative) "
                         "time between events [default: absolute]")
    ap.add_argument("-u", "--units", choices=["seconds", "ticks"],
                    default="seconds",
                    help="Time units to display [default: seconds]")
    ap.add_argument("-t", "--track", type=int, default=None,
                    help="Track number to inspect (0-based). "
                         "If omitted, all tracks are merged.")
    args = ap.parse_args()

    if not args.midi_file.exists():
        ap.error(f"File not found: {args.midi_file}")

    midi = mido.MidiFile(args.midi_file)
    print_midi(midi, args.track, args.mode, args.units)


if __name__ == "__main__":
    main()
