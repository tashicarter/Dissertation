import cv2
import os

def process_video(video_path, output_folder, frames_to_extract):
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the interval between frames to extract
    interval = max(total_frames // frames_to_extract, 1)

    frame_count = 0
    image_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video: {video_path}")
            break

        # Save the frame if it's within the interval
        if frame_count % interval == 0:
            image_count += 1
            image_path = os.path.join(output_folder, f"{os.path.basename(video_path)[:-4]}_frame_{image_count:04d}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved {image_path}")

        frame_count += 1

        # Stop if we've saved the required number of frames
        if image_count >= frames_to_extract:
            break

    cap.release()

def extract_images(video_folder, output_folder, frames_per_video):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each video
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        print(f"Processing video: {video_path}")
        process_video(video_path, output_folder, frames_per_video)

# Example usage
video_folder = "./videos"
output_folder = "output_images"
frames_per_video = 35

extract_images(video_folder, output_folder, frames_per_video)
