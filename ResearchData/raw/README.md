# raw/

Captures exactly as they were recorded. Never edited in place, never overwritten.

One subfolder per capture session, named `<source>_<date>` (for example `usrp-b210_20260914` or `gnuradio-sim_20260901`).
Each subfolder gets its own `README.md` describing how the data was produced:

```markdown
# usrp-b210_20260914

- **Source**: USRP B210, UHD 4.6
- **Centre frequency**: 868.1 MHz
- **Sample rate**: 1 Msps, complex float32 (`*.cf32`)
- **LoRa parameters**: SF7, BW 125 kHz, CR 4/5, preamble 8 symbols
- **Gain**: 40 dB RX
- **Duration**: 120 s
- **Environment**: indoor, ~5 m line of sight, two competing transmitters
- **Known issues**: clipping in the first ~3 s while AGC settled — skip it
```

Capture bytes are gitignored (`*.iq`, `*.cf32`, `*.sigmf-data`, `*.wav`, …). The README is not, and it is the only thing
that makes the capture interpretable later. Write it while the setup is still in front of you.

If a capture is genuinely irreplaceable, keep a backup outside this repository. Nothing here is a substitute for one.
