
def main():
    from get_frames import GetFrames
    from apply_filters import ApplyFilters
    from combine_frames import CombineFrames
    from save_difference import SaveDifference
    from visualization_histogram import VisualizationHistogram

    video_path = "cats.wmv"
    frames_folder = "frames"
    filtered_frames_folder = "filtered_frames"
    differences_folder = "differences"
    output_video_path = "output_video.avi"

    # Step 1: Extract frames
    frame_extractor = GetFrames(video_path)
    frame_extractor.extract_frames(frames_folder)

    # Step 2: Apply filters
    filter_applicator = ApplyFilters(frames_folder)
    filter_applicator.apply_gaussian_filter(filtered_frames_folder)

    # Step 3: Save differences between frames
    difference_saver = SaveDifference(filtered_frames_folder)
    difference_saver.save_frame_differences(differences_folder)

    # Step 4: Combine frames into video
    video_combiner = CombineFrames(filtered_frames_folder)
    video_combiner.combine_to_video(output_video_path, frame_rate=30)

    # Step 5: Plot compression efficiency chart
    VisualizationHistogram.plot_compression_chart(video_path, output_video_path)


if __name__ == "__main__":
    main()
