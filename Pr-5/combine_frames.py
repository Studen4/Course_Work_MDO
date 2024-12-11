class CombineFrames:
    def __init__(self, frames_folder):
        self.frames_folder = frames_folder

    def combine_to_video(self, output_video_path, frame_rate):
        import cv2
        import os

        frame_files = sorted(
            [f for f in os.listdir(self.frames_folder) if f.endswith(".png")]
        )

        first_frame = cv2.imread(os.path.join(self.frames_folder, frame_files[0]))
        height, width, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        for frame_file in frame_files:
            frame_path = os.path.join(self.frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            video.write(frame)

        video.release()
        print(f"Video saved to {output_video_path}")
        