# Using the voice in Microsoft Teams

```
Mic -> app -> "CABLE Input"  ==(VB-CABLE)==>  "CABLE Output" -> Teams microphone
```

1. Start the app (`run.ps1`).
2. **Microphone (input)**: your real microphone (never "CABLE Output").
3. **Output → Teams**: *CABLE Input (VB-Audio Virtual Cable)* (button **Use VB-CABLE**).
4. Optional **Monitor**: your **headphones**, to hear yourself. Speakers cause echo.
5. Press **Test beep → output**. In Windows *Sound settings → Input → CABLE Output* the
   level bar should move.
6. Choose **Robot voice v1** and press **Start**. Status turns blue: *Processing*.
7. Teams → *Settings → Devices*:
   * **Microphone: CABLE Output (VB-Audio Virtual Cable)**
   * Speaker: your headphones / speakers as usual (not CABLE)
   * **Noise suppression: Off or Low** (Teams' suppression can chop the robot buzz)
   * Use *Make a test call* to hear the result.
8. Tell meeting participants you are using a voice effect.

During a call you can switch device in Teams at any time back to your real microphone.

## New Teams vs. classic
Both: *Settings (…) → Settings → Devices → Audio devices → Microphone*. If CABLE Output
is missing, restart Teams after VB-CABLE was installed (and after the reboot).

## Latency
Measured on the development PC (i7-1165G7): processing ≈1 ms per 50 ms block, app
latency ≈137 ms + Windows device buffers (input reported 100 ms, output 22 ms), i.e. roughly
a quarter of a second end-to-end before Teams' own delay. The values for your PC are shown
live in the app.
