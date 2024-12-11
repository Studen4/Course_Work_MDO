class VisualizationHistogram:
    @staticmethod
    def plot_compression_chart(original_video_path, output_video_path):
        import os
        import matplotlib.pyplot as plt

        # Calculate sizes of videos
        original_size = os.path.getsize(original_video_path) / (1024 * 1024)  # in MB
        output_size = os.path.getsize(output_video_path) / (1024 * 1024)  # in MB

        categories = ['Original Video', 'Processed Video']
        sizes = [original_size, output_size]

        plt.bar(categories, sizes, color=['red', 'blue'])
        plt.title('Video Compression Efficiency')
        plt.ylabel('Size (MB)')
        plt.xlabel('Video Type')
        plt.show()
