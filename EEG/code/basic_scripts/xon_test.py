from pylsl import resolve_streams

print("Resolving all LSL streams (5 s timeout)...")
streams = resolve_streams(wait_time=5.0)

if not streams:
    print("No streams found.")
    quit()

for i, s in enumerate(streams):
    info = s
    print(f"[{i}] name={info.name()}, type={info.type()}, "
          f"source_id={info.source_id()}, nchans={info.channel_count()}")

