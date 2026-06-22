import argparse
import time
import csv
import pathlib
import numpy as np
import pygame
from pylsl import resolve_streams, StreamInlet, StreamInfo, StreamOutlet, local_clock

# ----------------------------------------------------------------------
# 1. Parse arguments
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pattern", type=str, default="A", choices=["A", "B"],
    help="Posture sequence pattern: A or B"
)
parser.add_argument(
    "--save-format", type=str, default="csv", choices=["npy", "csv"],
    help="Output format for saved data (npy or csv)"
)
parser.add_argument("--trials", type=int, default=80, help="Total trials to record")
parser.add_argument(
    "--output-dir",
    type=str,
    default="data/eeg_data",
    help="Directory where calibration CSV/NPY files are saved",
)
args = parser.parse_args()
pattern = args.pattern.upper()
save_format = args.save_format.lower()

print(f"Using posture pattern: {pattern}")
print(f"Saving format: {save_format}")

# ----------------------------------------------------------------------
# 2. Experiment settings
# ----------------------------------------------------------------------
N_TRIALS = args.trials

PROMPT_DURATION = 3.0     # s (text on screen)
FIXATION_DURATION = 1.0   # s (cross + 1 kHz)
POSTURE_DURATION = 3.0    # s (movement window, black screen)
END_DURATION = 0.50       # s (white circle + 50 Hz)

SCREEN_SIZE = (800, 600)
FONT_SIZE = 32

# Posture sequences
if pattern == "A":
    POSTURES = [
        "Flex your arm after the sound",
        "Rest your arm flexed after the sound",
        "Extend your arm after the sound",
        "Rest your arm extended after the sound"
    ]
else:
    POSTURES = [
        "Extend your arm after the sound",
        "Rest your arm extended after the sound",
        "Flex your arm after the sound",
        "Rest your arm flexed after the sound"
    ]

# ----------------------------------------------------------------------
# 3. Resolve X.on EEG stream and get time offset
# ----------------------------------------------------------------------
print("Resolving X.on EEG stream...")
streams = resolve_streams(wait_time=5.0)

eeg_info = None
for s in streams:
    if s.type() == "EEG" and s.name().startswith("X.on"):
        eeg_info = s
        break

if eeg_info is None:
    raise RuntimeError("No X.on EEG stream found. Make sure X.on app is streaming via LSL.")

n_chans = eeg_info.channel_count()
fs = eeg_info.nominal_srate() or 250.0
print(f"Using EEG stream: {eeg_info.name()}, chans={n_chans}, fs={fs}")

eeg_inlet = StreamInlet(eeg_info)

# Get time offset between our clock and EEG stream clock
time_offset = eeg_inlet.time_correction()
print(f"Time offset between local clock and EEG stream: {time_offset:.6f} seconds")

# ----------------------------------------------------------------------
# 4. Create marker stream
# ----------------------------------------------------------------------
marker_info = StreamInfo(
    name='XonMarkerStream',
    type='Markers',
    channel_count=1,
    nominal_srate=0,
    channel_format='string',
    source_id='xon_markers_01'
)
marker_outlet = StreamOutlet(marker_info)
print("Marker stream 'XonMarkerStream' created.")

# Data storage
eeg_data = []      # rows: [eeg_timestamp, ch1..chN]
marker_data = []   # rows: [local_timestamp, eeg_timestamp, label]

def send_marker(label: str):
    """Send marker and store with both local and EEG-synchronized timestamps."""
    local_ts = time.time()
    # Calculate the timestamp in EEG stream's time base
    eeg_sync_ts = local_clock() - time_offset
    marker_outlet.push_sample([label])
    marker_data.append([local_ts, eeg_sync_ts, label])
    print(f"Marker sent: {label} at local time {local_ts:.6f}, EEG time {eeg_sync_ts:.6f}")

# ----------------------------------------------------------------------
# 5. Pygame GUI + audio setup
# ----------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
font = pygame.font.SysFont(None, FONT_SIZE)
clock = pygame.time.Clock()
pygame.display.set_caption("X.on Psychophysics Protocol")

pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
AUDIO_FS = 44100

def make_tone(freq, duration, volume=0.4):
    t = np.linspace(0, duration, int(AUDIO_FS * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
    return pygame.mixer.Sound(wave.tobytes())

tone_fix = make_tone(1000, FIXATION_DURATION)
tone_end = make_tone(50, END_DURATION)

def draw_center_text(txt, y=0, color=(255, 255, 255)):
    surf = font.render(txt, True, color)
    rect = surf.get_rect(center=(SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2 + y))
    screen.blit(surf, rect)

def draw_cross():
    cx, cy = SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2
    pygame.draw.line(screen, (255, 255, 255), (cx - 40, cy), (cx + 40, cy), 4)
    pygame.draw.line(screen, (255, 255, 255), (cx, cy - 40), (cx, cy + 40), 4)

def draw_circle():
    cx, cy = SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2
    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 40, 5)

# ----------------------------------------------------------------------
# 6. Trial state machine
#    PROMPT -> FIXATION -> POSTURE -> END
# ----------------------------------------------------------------------
STATE_PROMPT = "prompt"
STATE_FIX = "fix"
STATE_POSTURE = "posture"
STATE_END = "end"

state = STATE_PROMPT
state_start = time.monotonic()
trial = 0

posture_text = POSTURES[trial % 4]

# base name for files
start_time_str = time.strftime("%Y%m%d_%H%M%S")
output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
base_name = str(output_dir / f"xon_{pattern}_{start_time_str}")

# fire entry markers once per trial, but only after we know EEG is coming in
prompt_entry_fired = False

print("Starting experiment... ESC to quit.")

while trial < N_TRIALS:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            trial = N_TRIALS
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            trial = N_TRIALS

    now = time.monotonic()
    elapsed = now - state_start

    # --- Pull EEG samples (time axis provided by LSL) ---
    chunk, ts = eeg_inlet.pull_chunk(timeout=0.0)
    if chunk and ts:
        for samp, tstamp in zip(chunk, ts):
            eeg_data.append([tstamp] + list(samp))

    # ---------------- Entry markers ----------------
    # We only send "Trial_X_Start / Posture_... / PromptStart" once,
    # when we are in PROMPT and have already started pulling EEG.
    if state == STATE_PROMPT and not prompt_entry_fired and len(eeg_data) > 0:
        send_marker(f"Trial_{trial+1}_Start")
        send_marker(f"Posture_{pattern}_{posture_text.replace(' ', '_')}")
        send_marker(f"PromptStart_Trial_{trial+1}")
        prompt_entry_fired = True

    # ---------------- State transitions + markers ----------------
    # PROMPT -> FIXATION
    if state == STATE_PROMPT and elapsed >= PROMPT_DURATION:
        state = STATE_FIX
        state_start = now
        tone_fix.play()
        send_marker(f"PromptEnd_Trial_{trial+1}")
        send_marker(f"FixationStart_Trial_{trial+1}")

    # FIXATION -> POSTURE
    elif state == STATE_FIX and elapsed >= FIXATION_DURATION:
        state = STATE_POSTURE
        state_start = now
        tone_fix.stop()
        send_marker(f"FixationEnd_Trial_{trial+1}")
        send_marker(f"PostureStart_Trial_{trial+1}")

    # POSTURE -> END
    elif state == STATE_POSTURE and elapsed >= POSTURE_DURATION:
        state = STATE_END
        state_start = now
        tone_end.play()
        send_marker(f"PostureEnd_Trial_{trial+1}")
        send_marker(f"TrialEnd_Trial_{trial+1}")

    # END -> next trial PROMPT or finish
    elif state == STATE_END and elapsed >= END_DURATION:
        send_marker(f"Trial_{trial+1}_Complete")
        trial += 1
        if trial >= N_TRIALS:
            break
        state = STATE_PROMPT
        state_start = now
        posture_text = POSTURES[trial % 4]
        prompt_entry_fired = False  # allow entry markers in next trial

    # ---------------- GUI drawing ----------------
    screen.fill((0, 0, 0))

    if state == STATE_PROMPT:
        draw_center_text(f"Trial {trial+1}/{N_TRIALS}", y=-80)
        draw_center_text(posture_text, y=0)

    elif state == STATE_FIX:
        draw_center_text(f"Trial {trial+1}/{N_TRIALS}", y=-120)
        draw_center_text("Fixation", y=-60)
        draw_cross()

    elif state == STATE_POSTURE:
        # black screen only during posture
        pass

    elif state == STATE_END:
        draw_center_text(f"Trial {trial+1}/{N_TRIALS}", y=-120)
        draw_center_text("End of trial", y=-60)
        draw_circle()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("Experiment finished. Saving data...")

# ----------------------------------------------------------------------
# 7. Save data with proper synchronization - FIXED VERSION
# ----------------------------------------------------------------------
eeg_arr = np.asarray(eeg_data, dtype=np.float64) if eeg_data else np.empty((0, 1 + n_chans))

print(f"Recorded {eeg_arr.shape[0]} EEG samples, {len(marker_data)} markers.")

if save_format == "npy":
    np.save(base_name + "_eeg.npy", eeg_arr)
    np.save(base_name + "_markers.npy", np.array(marker_data, dtype=object))
    print(f"Saved EEG to {base_name}_eeg.npy")
    print(f"Saved markers to {base_name}_markers.npy")

elif save_format == "csv":
    # Save separate files first
    if eeg_arr.shape[0] > 0:
        with open(base_name + "_eeg.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["eeg_timestamp"] + [f"ch{i+1}" for i in range(n_chans)])
            for row in eeg_arr:
                writer.writerow(row)
        print(f"Saved EEG to {base_name}_eeg.csv")
    
    if marker_data:
        with open(base_name + "_markers.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["local_timestamp", "eeg_timestamp", "label"])
            for local_ts, eeg_ts, label in marker_data:
                writer.writerow([local_ts, eeg_ts, label])
        print(f"Saved markers to {base_name}_markers.csv")
    
    # Create merged file with proper marker assignment - FIXED ALGORITHM
    if eeg_arr.shape[0] > 0 and marker_data:
        eeg_timestamps = eeg_arr[:, 0]
        
        # Sort both by EEG timestamp
        eeg_arr = eeg_arr[eeg_timestamps.argsort()]
        eeg_timestamps = eeg_arr[:, 0]
        
        # Filter markers - ONLY KEEP PostureStart markers
        filtered_markers = []
        for local_ts, eeg_ts, label in marker_data:
            if "PostureStart" in label:
                filtered_markers.append((eeg_ts, label))
        
        print(f"Filtered to {len(filtered_markers)} PostureStart markers")
        
        # Create merged data with empty markers
        merged_data = [list(eeg_row) + [""] for eeg_row in eeg_arr]
        
        # Assign markers one per row, distributing to subsequent rows if needed
        marker_queue = filtered_markers.copy()
        current_marker_index = 0
        
        for i, eeg_row in enumerate(eeg_arr):
            current_eeg_time = eeg_row[0]
            
            # If we have markers in queue and this row doesn't have a marker yet
            if current_marker_index < len(marker_queue) and not merged_data[i][-1]:
                marker_eeg_ts, marker_label = marker_queue[current_marker_index]
                
                # Check if this marker is close enough to current EEG sample
                time_diff = abs(marker_eeg_ts - current_eeg_time)
                if time_diff < 0.1:  # 100ms window
                    merged_data[i][-1] = marker_label
                    current_marker_index += 1
                    print(f"Assigned marker '{marker_label}' to EEG sample at {current_eeg_time:.6f} (diff: {time_diff:.6f}s)")
        
        # Save merged file - ensure markers go in ONE column
        with open(base_name + "_merged.csv", "w", newline="") as f:
            writer = csv.writer(f)
            # Header with exactly n_chans + 2 columns (timestamp + n_chans + markers)
            writer.writerow(["eeg_timestamp"] + [f"ch{i+1}" for i in range(n_chans)] + ["markers"])
            for row in merged_data:
                writer.writerow(row)
        
        print(f"Saved merged data to {base_name}_merged.csv")
        
        # Count assigned markers
        assigned_count = sum(1 for row in merged_data if row[-1])
        print(f"PostureStart markers assigned: {assigned_count} out of {len(filtered_markers)}")

print("Data saving complete!")
