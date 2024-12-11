class ApplyFilters:
    def __init__(self, frames_folder):
        self.frames_folder = frames_folder

    def apply_gaussian_filter(self, output_folder):
        import cv2
        import os

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for frame_file in os.listdir(self.frames_folder):
            frame_path = os.path.join(self.frames_folder, frame_file)
            frame = cv2.imread(frame_path)

            filtered_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            output_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_path, filtered_frame)

        print(f"Filtered frames saved to {output_folder}")
