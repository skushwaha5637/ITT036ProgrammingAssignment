#!/usr/bin/env python3
"""ITT 036 - Programming Assignment 1: digital signal encoder / decoder.

Run:  python signal_codec.py            (interactive)
      python signal_codec.py --selftest (round-trip tests of every scheme)
      python signal_codec.py --demo DIR (save sample plots into DIR, no GUI)

Waveforms are sampled arrays with SPB = 2 samples per bit (half-bit
resolution, needed for Manchester). Decoders only look at these sampled
levels (the signal "image"), never at the original bit list.
"""
import argparse
import os
import random

import numpy as np

SPB = 2          # samples per bit
INIT = -1        # line level before the first bit (NRZ-I, Differential Manchester)


# --------------------------------------------------------------------------
# Digital data generator
# --------------------------------------------------------------------------
def random_bits(n, rng=random):
    return [rng.randint(0, 1) for _ in range(n)]


def bits_with_patterns(n, rng=random):
    """Random bits with fixed sub-sequences of 8 and 4 consecutive zeros."""
    bits = random_bits(n, rng)
    for run in (8, 4, 8, 4):
        if n >= run:
            p = rng.randrange(0, n - run + 1)
            bits[p:p + run] = [0] * run
    return bits


def longest_palindrome(s):
    """Manacher's algorithm, O(n). Returns (palindrome, start_index)."""
    if not s:
        return "", 0
    t = "^#" + "#".join(s) + "#$"
    p = [0] * len(t)
    c = r = 0
    for i in range(1, len(t) - 1):
        if i < r:
            p[i] = min(r - i, p[2 * c - i])
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]
    m = max(range(len(p)), key=p.__getitem__)
    start = (m - p[m]) // 2
    return s[start:start + p[m]], start


# --------------------------------------------------------------------------
# Line coding encoders  (bits -> waveform of +1 / 0 / -1)
# --------------------------------------------------------------------------
def _hold(levels):
    return np.repeat(np.asarray(levels), SPB)


def _halves(first, second):
    return np.stack([first, second], axis=1).ravel()


def nrz_l(bits):                       # 0 -> high, 1 -> low
    return _hold(np.where(np.asarray(bits) == 0, 1, -1))


def nrz_i(bits):                       # 1 -> invert at start of bit
    inv = np.where(np.asarray(bits) == 1, -1, 1)
    return _hold(INIT * np.cumprod(inv))


def manchester(bits):                  # 1 -> low-to-high, 0 -> high-to-low
    first = np.where(np.asarray(bits) == 1, -1, 1)
    return _halves(first, -first)


def diff_manchester(bits):             # mid-bit transition always; 0 -> extra transition at start
    flip = np.where(np.asarray(bits) == 1, 1, -1)       # 1 keeps level, 0 inverts
    start = np.empty(len(flip), dtype=int)
    prev = INIT
    for i, f in enumerate(flip):
        start[i] = prev * f
        prev = -start[i]
    return _halves(start, -start)


def ami(bits):                         # 1 -> alternating +/-, 0 -> 0
    b = np.asarray(bits)
    pol = np.where(np.cumsum(b) % 2 == 1, 1, -1)
    return _hold(b * pol)


def _zero_run_end(bits, i):
    j = i
    while j < len(bits) and bits[j] == 0:
        j += 1
    return j


def b8zs(bits):
    """AMI + replace 8 zeros by 000VB0VB. Single linear pass."""
    n = len(bits)
    out = [0] * n
    last, i = -1, 0
    while i < n:
        if bits[i]:
            last = -last
            out[i] = last
            i += 1
            continue
        j = _zero_run_end(bits, i)
        k, run = i, j - i
        while run >= 8:
            out[k + 3], out[k + 4] = last, -last
            out[k + 6], out[k + 7] = -last, last       # polarity ends equal to `last`
            k += 8
            run -= 8
        i = j
    return _hold(out)


def hdb3(bits):
    """AMI + replace 4 zeros by 000V or B00V (parity of pulses since last substitution)."""
    n = len(bits)
    out = [0] * n
    last, pulses, i = -1, 0, 0
    while i < n:
        if bits[i]:
            last = -last
            out[i] = last
            pulses += 1
            i += 1
            continue
        j = _zero_run_end(bits, i)
        k, run = i, j - i
        while run >= 4:
            if pulses % 2 == 0:                          # even -> B00V
                last = -last
                out[k] = out[k + 3] = last
            else:                                        # odd  -> 000V
                out[k + 3] = last
            pulses = 0
            k += 4
            run -= 4
        i = j
    return _hold(out)


# --------------------------------------------------------------------------
# Line coding decoders  (waveform samples -> bits)
# --------------------------------------------------------------------------
def _first(sig):
    return np.asarray(sig)[0::SPB]


def _second(sig):
    return np.asarray(sig)[1::SPB]


def nrz_l_dec(sig):
    return (_first(sig) == -1).astype(int).tolist()


def nrz_i_dec(sig):
    lv = _first(sig)
    prev = np.concatenate([[INIT], lv[:-1]])
    return (lv != prev).astype(int).tolist()


def manchester_dec(sig):
    return (_second(sig) > _first(sig)).astype(int).tolist()


def diff_manchester_dec(sig):
    prev_end = np.concatenate([[INIT], _second(sig)[:-1]])
    return (_first(sig) == prev_end).astype(int).tolist()


def ami_dec(sig):
    return (_first(sig) != 0).astype(int).tolist()


def b8zs_dec(sig):
    p = _first(sig).tolist()
    n = len(p)
    bits = [0] * n
    last, i = -1, 0
    while i < n:
        if p[i] != 0:
            bits[i] = 1
            last = p[i]
            i += 1
        elif p[i:i + 8] == [0, 0, 0, last, -last, 0, -last, last]:
            i += 8                                       # substitution -> eight zeros
        else:
            i += 1
    return bits


def hdb3_dec(sig):
    p = _first(sig).tolist()
    bits = [int(v != 0) for v in p]
    last = 0
    for i, v in enumerate(p):
        if v == 0:
            continue
        if v == last:                                    # violation pulse V
            bits[max(0, i - 3):i + 1] = [0] * (min(i, 3) + 1)
        else:
            last = v
    return bits


# Registries (dictionary dispatch - no if/else chains)
LINE_CODES = {
    "1": ("NRZ-L", nrz_l, nrz_l_dec),
    "2": ("NRZ-I", nrz_i, nrz_i_dec),
    "3": ("Manchester", manchester, manchester_dec),
    "4": ("Differential Manchester", diff_manchester, diff_manchester_dec),
    "5": ("AMI", ami, ami_dec),
}
SCRAMBLING = {
    "1": ("B8ZS", b8zs, b8zs_dec),
    "2": ("HDB3", hdb3, hdb3_dec),
}
ALL_SCHEMES = [LINE_CODES[k] for k in LINE_CODES] + [SCRAMBLING[k] for k in SCRAMBLING]


# --------------------------------------------------------------------------
# PCM and Delta modulation
# --------------------------------------------------------------------------
def pcm_encode(x, nbits, vmax):
    levels = 2 ** nbits
    q = np.clip(np.round((np.asarray(x) + vmax) / (2 * vmax) * (levels - 1)), 0, levels - 1).astype(int)
    shifts = np.arange(nbits - 1, -1, -1)
    bits = ((q[:, None] >> shifts) & 1).ravel().tolist()
    return bits, q


def pcm_decode(bits, nbits, vmax):
    levels = 2 ** nbits
    q = np.asarray(bits).reshape(-1, nbits) @ (1 << np.arange(nbits - 1, -1, -1))
    return q / (levels - 1) * 2 * vmax - vmax, q


def dm_encode(x, delta, start=0.0):
    s, bits, stair = start, [], []
    for v in x:
        b = int(v > s)
        s += delta if b else -delta
        bits.append(b)
        stair.append(s)
    return bits, np.array(stair)


def dm_decode(bits, delta, start=0.0):
    return start + np.cumsum(np.where(np.asarray(bits) == 1, delta, -delta))


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _plt(save_dir):
    import matplotlib
    if save_dir:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_signal(ax, bits, sig, title):
    t = np.arange(len(sig) + 1) / SPB
    ax.step(t, np.append(sig, sig[-1]), where="post", lw=2)
    ax.set_ylim(-1.8, 1.8)
    ax.set_yticks([-1, 0, 1])
    ax.set_xlim(0, len(bits))
    for i, b in enumerate(bits):
        ax.axvline(i, color="gray", ls=":", lw=0.6)
        ax.text(i + 0.5, 1.5, str(b), ha="center", fontsize=8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def show_signals(panels, save_dir=None, name="signal"):
    plt = _plt(save_dir)
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(8, len(panels[0][0]) * 0.45), 2.6 * len(panels)),
                             squeeze=False)
    for ax, (bits, sig, title) in zip(axes[:, 0], panels):
        draw_signal(ax, bits, sig, title)
    axes[-1, 0].set_xlabel("bit interval")
    fig.tight_layout()
    finish(plt, fig, save_dir, name)


def finish(plt, fig, save_dir, name):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, name + ".png"), dpi=120)
        plt.close(fig)
    else:
        plt.show()


def show_analog(t, x, recon, label, save_dir=None, name="analog"):
    plt = _plt(save_dir)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, x, label="analog input")
    ax.step(t, recon, where="post", label=label)
    ax.set_xlabel("time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    finish(plt, fig, save_dir, name)


# --------------------------------------------------------------------------
# Interactive helpers
# --------------------------------------------------------------------------
def ask(prompt, options):
    """options: dict key -> label. Re-asks until valid."""
    while True:
        print(prompt)
        for k, v in options.items():
            print(f"  {k}) {v}")
        ans = input("> ").strip().lower()
        if ans in options:
            return ans
        print("Invalid choice, try again.\n")


def ask_float(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return float(raw) if raw else float(default)


def ask_int(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw else int(default)


def yes(prompt):
    return input(prompt + " (y/n): ").strip().lower().startswith("y")


def get_digital_bits():
    kind = ask("Digital data source:", {"1": "completely random", "2": "random with 8/4 consecutive zeros",
                                         "3": "type my own bits"})
    typed = lambda: [int(c) for c in input("Bits (e.g. 1010000000011): ").strip() if c in "01"]
    gens = {
        "1": lambda n: random_bits(n),
        "2": lambda n: bits_with_patterns(n),
        "3": lambda n: typed(),
    }
    n = ask_int("Length (ignored for typed input)", 32) if kind != "3" else 0
    return gens[kind](n)


def choose_scheme():
    key = ask("Line coding scheme:", {k: v[0] for k, v in LINE_CODES.items()})
    name, enc, dec = LINE_CODES[key]
    steps = {"AMI": scrambling_step}
    return steps.get(name, lambda *a: (name, enc, dec, None))(name, enc, dec)


def scrambling_step(name, enc, dec):
    if not yes("AMI selected. Is scrambling needed?"):
        return name, enc, dec, None
    key = ask("Scrambling type:", {k: v[0] for k, v in SCRAMBLING.items()})
    sname, senc, sdec = SCRAMBLING[key]
    return sname, senc, sdec, (name, enc)          # also keep plain AMI for comparison


def encode_and_report(bits, save_dir=None):
    pal, pos = longest_palindrome("".join(map(str, bits)))
    print(f"\nDigital data  : {''.join(map(str, bits))}")
    print(f"Longest palindrome: {pal!r} (length {len(pal)}, starts at index {pos})")
    name, enc, dec, base = choose_scheme()
    sig = enc(bits)
    panels = [(bits, sig, f"{name} encoded signal")]
    if base:
        panels.insert(0, (bits, base[1](bits), f"{base[0]} (before scrambling)"))
    print(f"\nEncoded with {name}. Signal levels (2 samples/bit):\n{sig.tolist()}")
    show_signals(panels, save_dir, name.replace(" ", "_"))
    if yes("\nDo you want to decode the signal?"):
        out = dec(sig)                         # decoder sees only the waveform
        print("Decoded bits :", "".join(map(str, out)))
        print("Matches input:", out == list(bits))


def analog_input():
    kind = ask("Analog input:", {"1": "sine wave", "2": "type sample values"})
    if kind == "1":
        amp = ask_float("Amplitude", 1)
        freq = ask_float("Frequency (Hz)", 1)
        fs = ask_float("Sampling rate (Hz)", 16)
        dur = ask_float("Duration (s)", 2)
        t = np.arange(0, dur, 1 / fs)
        return t, amp * np.sin(2 * np.pi * freq * t), amp
    x = np.array([float(v) for v in input("Comma separated samples: ").split(",")])
    return np.arange(len(x)), x, float(np.max(np.abs(x))) or 1.0


def analog_flow(save_dir=None):
    t, x, vmax = analog_input()
    tech = ask("Technique:", {"1": "PCM", "2": "Delta modulation (DM)"})
    if tech == "1":
        nb = ask_int("Bits per sample", 3)
        bits, q = pcm_encode(x, nb, vmax)
        recon, _ = pcm_decode(bits, nb, vmax)
        label = "PCM decoded staircase"
        print("Quantised levels:", q.tolist())
    else:
        delta = ask_float("Step size delta", round(vmax / 8, 4))
        bits, stair = dm_encode(x, delta)
        recon = dm_decode(bits, delta)
        label = "DM decoded staircase"
    print("Digital data:", "".join(map(str, bits)))
    show_analog(t, x, recon, label, save_dir, "pcm_dm")
    if yes("Feed the digital data to a line coding scheme?"):
        encode_and_report(bits, save_dir)


def main_interactive(save_dir=None):
    src = ask("Input type:", {"1": "digital", "2": "analog"})
    flows = {"1": lambda: encode_and_report(get_digital_bits(), save_dir), "2": lambda: analog_flow(save_dir)}
    flows[src]()


# --------------------------------------------------------------------------
# Self test / demo
# --------------------------------------------------------------------------
def selftest():
    rng = random.Random(7)
    cases = [[0] * 30, [1] * 10, [0] * 8, [0] * 4, [1, 0, 0, 0, 0, 1], [0] * 17 + [1] + [0] * 9]
    cases += [random_bits(rng.randint(1, 80), rng) for _ in range(300)]
    cases += [bits_with_patterns(rng.randint(12, 80), rng) for _ in range(300)]
    for name, enc, dec in ALL_SCHEMES:
        for bits in cases:
            out = dec(enc(bits))
            assert out == list(bits), f"{name} failed on {bits}: {out}"
        print(f"{name:25s} OK ({len(cases)} cases)")
    # scrambling properties: never 4 (HDB3) / 8 (B8ZS) consecutive zero levels
    for bits in cases:
        for enc, k in ((hdb3, 4), (b8zs, 8)):
            lv = enc(bits)[::SPB].tolist()
            run = best = 0
            for v in lv:
                run = run + 1 if v == 0 else 0
                best = max(best, run)
            assert best < k, (enc.__name__, bits)
    # palindrome vs brute force
    for _ in range(200):
        s = "".join(map(str, random_bits(rng.randint(1, 30), rng)))
        brute = max((s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1) if s[i:j] == s[i:j][::-1]),
                    key=len)
        assert len(longest_palindrome(s)[0]) == len(brute)
    print("Palindrome (Manacher) OK")
    x = 0.9 * np.sin(np.linspace(0, 6.28, 40))
    bits, _ = pcm_encode(x, 4, 1.0)
    assert np.max(np.abs(pcm_decode(bits, 4, 1.0)[0] - x)) < 1 / 15 + 1e-9
    bits, st = dm_encode(x, 0.2)
    assert np.allclose(dm_decode(bits, 0.2), st)
    print("PCM / DM OK\nALL TESTS PASSED")


def demo(d):
    rng = random.Random(3)
    bits = bits_with_patterns(24, rng)
    bits[6:14] = [0] * 8
    for name, enc, _ in ALL_SCHEMES:
        show_signals([(bits, enc(bits), name)], d, name.replace(" ", "_"))
    show_signals([(bits, ami(bits), "AMI"), (bits, b8zs(bits), "B8ZS"), (bits, hdb3(bits), "HDB3")], d, "scrambling")
    t = np.arange(0, 2, 1 / 16)
    x = np.sin(2 * np.pi * t)
    b, _ = pcm_encode(x, 3, 1.0)
    show_analog(t, x, pcm_decode(b, 3, 1.0)[0], "PCM staircase", d, "pcm")
    b, _ = dm_encode(x, 0.25)
    show_analog(t, x, dm_decode(b, 0.25), "DM staircase", d, "dm")
    print("saved plots to", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", metavar="DIR")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.demo:
        demo(a.demo)
    else:
        main_interactive()
