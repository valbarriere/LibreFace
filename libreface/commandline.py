import argparse

from libreface import get_facial_attributes

def main_func():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True, help = "Path to the video or image which you want to process through libreface")
    parser.add_argument("--output_path", type=str, default="sample_results.csv", help="Path to the csv where results should be saved. Defaults to 'sample_results.csv'")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use while inference. Can be 'cpu', 'cuda:0', 'cuda:1', ... Defaults to 'cpu'")
    parser.add_argument("--temp", type=str, default="./tmp", help="Path where the temporary results for facial attributes can be saved.")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of frames to process in a single batch when doing inference on a video.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers to be used in the dataloader while doing inference on a video.")
    parser.add_argument("--frame_interval", type=int, default=1, help="Number of frame_interval to skip (eq. to change fps).")
    parser.add_argument("--remove_temp", action='store_true', help="Remove the frames in the temp video folder (save memory)")
    parser.add_argument("--path_scene_cuts", type=str, default="", help="If the file is long and should be cut into parts.")

    args = parser.parse_args()

    get_facial_attributes(args.input_path, 
                          output_save_path=args.output_path, 
                          model_choice="joint_au_detection_intensity_estimator",
                          temp_dir=args.temp, 
                          device=args.device,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          frame_interval=args.frame_interval,
                          remove_temp=args.remove_temp,
                          path_scene_cuts=args.path_scene_cuts)

if __name__ ==  "__main__":
    main_func()
