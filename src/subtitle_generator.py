from src.config import Config as cfg

def generate_srt(word_timestamps, output_path, frame_rate=100):
    """
    word_timestamps: list of (word, start_frame, end_frame)
    frame_rate: frames per second (10ms = 100 fps)
    """
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_path, 'w') as f:
        idx = 1
        for word, start_frame, end_frame in word_timestamps:
            start_sec = start_frame / frame_rate
            end_sec = end_frame / frame_rate
            if start_sec >= end_sec:
                end_sec = start_sec + 0.1
            f.write(f"{idx}\n{fmt(start_sec)} --> {fmt(end_sec)}\n{word}\n\n")
            idx += 1