import cv2
import numpy as np
import re

class PianoKeyboardCV:
    """
    Draw an 88-key (or shorter) piano keyboard with OpenCV.
    Each key can be recoloured individually by MIDI number or note name.
    """
    # ── geometry (pixels) ────────────────────────────────────────────────
    WHITE_W, WHITE_H = 20, 100        # single white key
    BLACK_W, BLACK_H = 12, 65        # single black key


    OUTLINE_THICKNESS = 1

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

    # ── construction ─────────────────────────────────────────────────────
    def __init__(self, start_midi: int = 21, num_keys: int = 87):
        self.start = start_midi
        self.notes = list(range(start_midi, start_midi + num_keys))

        # per-note colour state (BGR tuples) – default: white / black
        self.colours = {
            n: (255, 255, 255) if self._is_white(n) else (0, 0, 0)
            for n in self.notes
        }

        # x-position of each white key (index * WHITE_W)
        self._white_x = {}
        x = 0
        for n in self.notes:
            if self._is_white(n):
                self._white_x[n] = x
                x += self.WHITE_W

        self.width  = x
        self.height = self.WHITE_H
        self._render()                       # create self.img

    # ── public API ───────────────────────────────────────────────────────
    def color_keys(self, mapping: dict):
        """
        mapping: {MIDI int | note str: colour}
                 colour = tuple/list of 3 ints (B, G, R) or any OpenCV colour name
        """
        for key, colour in mapping.items():
            midi = self._to_midi(key)
            if midi in self.colours:
                self.colours[midi] = self._parse_colour(colour)

        self._render()

    def show(self, winname='Piano', wait=True):
        cv2.imshow(winname, self.img)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save(self, filename):
        cv2.imwrite(filename, self.img)

    # ── internal helpers ─────────────────────────────────────────────────
    def _render(self):
        # blank white canvas
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        # Pass 1 – draw white keys
        for n in self.notes:
            if self._is_white(n):
                x = self._white_x[n]
                cv2.rectangle(
                    canvas,
                    (x, 0),
                    (x + self.WHITE_W, self.WHITE_H),
                    self.colours[n],
                    thickness=-1,
                )
                cv2.rectangle(
                    canvas,
                    (x, 0),
                    (x + self.WHITE_W, self.WHITE_H),
                    (0, 0, 0),
                    thickness=self.OUTLINE_THICKNESS,
                )

        # Pass 2 – draw black keys on top
        for n in self.notes:
            if not self._is_white(n):
                # preceding white key sets the anchor
                prev_white = n - 1
                while prev_white not in self._white_x:
                    prev_white -= 1
                x_left = self._white_x[prev_white] + self.WHITE_W - self.BLACK_W // 2
                cv2.rectangle(
                    canvas,
                    (x_left, 0),
                    (x_left + self.BLACK_W, self.BLACK_H),
                    self.colours[n],
                    thickness=-1,
                )
                cv2.rectangle(
                    canvas,
                    (x_left, 0),
                    (x_left + self.BLACK_W, self.BLACK_H),
                    (0, 0, 0),
                    thickness=self.OUTLINE_THICKNESS,
                )

        self.img = canvas

    @staticmethod
    def _is_white(midi):               # C D E F G A B
        return (midi % 12) in PianoKeyboardCV._WHITE_SET

    def _to_midi(self, key):
        if isinstance(key, int):
            return key
        m = self._NOTE_RE.fullmatch(key.strip())
        if not m:
            raise ValueError(f"Bad note name: {key!r}")
        base, octave = m.groups()
        base = base.capitalize()
        octave = int(octave)
        if base not in self._NOTE_BASES:
            raise ValueError(f"Unknown pitch class: {base}")
        return self._NOTE_BASES[base] + (octave + 1) * 12

    @staticmethod
    def _parse_colour(c):
        """Accept BGR tuple/list, RGB tuple/list, hex '#RRGGBB', or named colour."""
        if isinstance(c, (tuple, list)):
            if len(c) != 3:
                raise ValueError("Colour must have 3 components")
            return tuple(int(x) for x in c[::-1])   # assume RGB → BGR
        if isinstance(c, str):
            c = c.strip()
            if c.startswith('#') and len(c) == 7:
                r = int(c[1:3], 16)
                g = int(c[3:5], 16)
                b = int(c[5:7], 16)
                return (b, g, r)
            # basic OpenCV colour names
            named = {
                'black': (0, 0, 0),
                'white': (255, 255, 255),
                'red':   (0, 0, 255),
                'green': (0, 255, 0),
                'blue':  (255, 0, 0),
                'yellow':(0, 255, 255),
                'cyan':  (255, 255, 0),
                'magenta':(255, 0, 255),
                'grey':  (128, 128, 128),
                'gray':  (128, 128, 128),
            }
            if c.lower() in named:
                return named[c.lower()]
        raise ValueError(f"Unsupported colour: {c!r}")

# ── quick demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    kb = PianoKeyboardCV()                       # default 88-key layout

    # recolour three keys: C4 = red, E4 = blue, G4 (MIDI 67) = lime green
    kb.color_keys({
        'C4':  'red',
        'E4':  (0, 0, 255),   # OpenCV uses BGR – (B,G,R) → pure red
        67:    '#32CD32',     # hex for lime green
    })

    kb.show()                # or kb.save('piano.png')
