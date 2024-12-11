
class SaveDifference:
    def __init__(self, frames_folder):
        self.frames_folder = frames_folder

    def save_frame_differences(self, output_folder):
        import cv2
        import os
        import numpy as np

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        frame_files = sorted(os.listdir(self.frames_folder))
        previous_frame = None

        for frame_file in frame_files:
            frame_path = os.path.join(self.frames_folder, frame_file)
            current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

            if previous_frame is not None:
                diff = cv2.absdiff(current_frame, previous_frame)
                diff_path = os.path.join(output_folder, f"diff_{frame_file}")
                cv2.imwrite(diff_path, diff)

            previous_frame = current_frame

        print(f"Frame differences saved to {output_folder}")
