

class GetFrames:
    def __init__(self, video_path):
        self.video_path = video_path

    def extract_frames(self, output_folder):
        import cv2
        import os

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        video = cv2.VideoCapture(self.video_path)
        frame_number = 0

        while True:
            success, frame = video.read()
            if not success:
                break

            frame_path = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
            cv2.imwrite(frame_path, frame)
            frame_number += 1

        video.release()
        print(f"Frames saved to {output_folder}")