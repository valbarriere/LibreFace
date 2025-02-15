import pandas as pd
import time
from tqdm import tqdm
import warnings
import shutil
import glob
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
pd.options.mode.chained_assignment = None  # default='warn'

from libreface.detect_mediapipe_image import *
from libreface.AU_Detection.inference import detect_action_units, detect_action_units_video
from libreface.AU_Recognition.inference import get_au_intensities, get_au_intensities_and_detect_aus, get_au_intensities_and_detect_aus_video, get_au_intensities_video
from libreface.Facial_Expression_Recognition.inference import get_facial_expression, get_facial_expression_video
from libreface.utils import get_frames_from_video_ffmpeg, uniquify_file, check_file_type

def get_facial_attributes_image(image_path:str, 
                                model_choice:str="joint_au_detection_intensity_estimator", 
                                temp_dir:str="./tmp", 
                                device:str="cpu",
                                weights_download_dir:str = "./weights_libreface")->dict:
    """Get facial attributes for an image. This function reads an image and returns a dictionary containing
    some detected facial action units and expressions.

    Args:
        image_path (str): Input image path.
        model_choice (str, optional): Model to use when doing predictions. Defaults to "joint_au_detection_intensity_estimator".
        temp_dir (str, optional): Path where the temporary aligned image, facial landmarks 
        and landmark annotated image will be stored. Defaults to "./tmp".
        device (str, optional): device to be used for inference. Can be "cpu" or "cuda". Defaults to "cpu".
        weights_download_dir(str, optional): directory where you want to download and save the model weights.

    Returns:
        dict: dictionary containing the following keys
            input_image_path - copied from the image_path
            aligned_image_path - path to the aligned image, i.e. image with the only the face of the person cropped from original image
            detected_action_units - dictionary of detected action units. Units which are detected have value of 1, else 0.
            au_intensities - dictionary of action unit intensities, with each intensity in the range (0, 5)
            facial_expression - detected facial expression. Can be one from ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    """
    
    print(f"Using device: {device} for inference...")
    aligned_image_path, headpose, landmarks_2d, is_zoom = get_aligned_image(image_path, temp_dir=temp_dir, verbose=False)
    if model_choice == "joint_au_detection_intensity_estimator":
        detected_aus, au_intensities = get_au_intensities_and_detect_aus(aligned_image_path, device=device, weights_download_dir=weights_download_dir)
    elif model_choice == "separate_prediction_heads":
        detected_aus = detect_action_units(aligned_image_path, device = device, weights_download_dir=weights_download_dir)
        au_intensities = get_au_intensities(aligned_image_path, device = device, weights_download_dir=weights_download_dir)
    else:
        print(f"Undefined model_choice = {model_choice} for get_facial_attributes_image()")
        raise NotImplementedError
    facial_expression = get_facial_expression(aligned_image_path, device = device, weights_download_dir=weights_download_dir)
    return_dict =  {
            "detected_aus": detected_aus,
            "au_intensities": au_intensities,
            "facial_expression": facial_expression}
    
    return_dict = {**return_dict, **headpose, **landmarks_2d}

    return return_dict

def get_facial_attributes_video_ini(video_path, 
                                model_choice:str="joint_au_detection_intensity_estimator", 
                                temp_dir="./tmp", 
                                device="cpu",
                                batch_size = 256,
                                num_workers = 2,
                                weights_download_dir:str = "./weights_libreface", 
                                frame_interval=1,
                                remove_temp=False):
    print(f"Using device: {device} for inference...")
    
    frames_df = get_frames_from_video_ffmpeg(video_path, temp_dir=temp_dir, frame_interval=frame_interval)
    cur_video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    aligned_frames_path_list, headpose_list, landmarks_3d_list, is_zoom_list = get_aligned_video_frames(frames_df, temp_dir=os.path.join(temp_dir, cur_video_name))
    # frames_df["aligned_frame_path"] = aligned_frames_path_list
    frames_df = frames_df.drop("path_to_frame", axis=1)
    frames_df["headpose"] = headpose_list
    frames_df["landmarks_3d"] = landmarks_3d_list
     

    frames_df = frames_df.join(pd.json_normalize(frames_df['headpose'])).drop('headpose', axis='columns')
    frames_df = frames_df.join(pd.json_normalize(frames_df['landmarks_3d'])).drop('landmarks_3d', axis='columns')

    detected_aus, au_intensities, facial_expression = [], [], []
    
    if model_choice == "joint_au_detection_intensity_estimator":
        detected_aus, au_intensities = get_au_intensities_and_detect_aus_video(aligned_frames_path_list, 
                                                                           device = device, 
                                                                           batch_size=batch_size,
                                                                           num_workers=num_workers,
                                                                           weights_download_dir=weights_download_dir)
    elif model_choice == "separate_prediction_heads":
        detected_aus = detect_action_units_video(aligned_frames_path_list, 
                                                device=device,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                weights_download_dir=weights_download_dir)
        au_intensities = get_au_intensities_video(aligned_frames_path_list, 
                                                device=device,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                weights_download_dir=weights_download_dir)
    else:
        print(f"Undefined model_choice = {model_choice} for get_facial_attributes_video()")
        raise NotImplementedError
    
    facial_expression = get_facial_expression_video(aligned_frames_path_list, 
                                                    device = device, 
                                                    batch_size=batch_size,
                                                    weights_download_dir=weights_download_dir)
    

    frames_df = frames_df.join(detected_aus)
    frames_df = frames_df.join(au_intensities)
    frames_df = frames_df.join(facial_expression)
    if remove_temp:
        temp_dir = os.path.join(temp_dir, cur_video_name)
        print('Remove temp dir', temp_dir)
        shutil.rmtree(temp_dir)
    return frames_df

def convert2s(ds):
    return pd.to_timedelta(ds).dt.total_seconds()

def get_facial_attributes_video(video_path, 
                                model_choice:str="joint_au_detection_intensity_estimator", 
                                temp_dir="./tmp", 
                                device="cpu",
                                batch_size = 256,
                                num_workers = 2,
                                weights_download_dir:str = "./weights_libreface", 
                                frame_interval=1,
                                remove_temp=False,
                                path_scene_cuts="/data/vbarrier/standup/videos/scenes/fr.csv",
                                verbose=False):
    
    print(f"Using device: {device} for inference...")
    
    # Return a list of scene
    all_frames_df = get_frames_from_video_ffmpeg(video_path, temp_dir=temp_dir, frame_interval=frame_interval)

    cur_video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])

    if path_scene_cuts:
        dfscene = pd.read_csv(path_scene_cuts)
        cur_video_name_ext = video_path.split("/")[-1]
        dfscene = dfscene[dfscene.id_video == cur_video_name_ext]
        list_scenes = dfscene.id_scene.values
        dfscene.timecode_start = convert2s(dfscene.timecode_start)
        dfscene.timecode_end = convert2s(dfscene.timecode_end)
    else:
        list_scenes = [0]

    list_frames_df_processed = []
    
    for id_scene in tqdm(list_scenes, desc="Processing video scene by scene..."):
        
        if path_scene_cuts:
            # get only the part of the videos for one scene
            tst, tend = dfscene.iloc[id_scene][['timecode_start', 'timecode_end']].values
            frames_df = all_frames_df[(all_frames_df.frame_time_in_ms >=tst) &(all_frames_df.frame_time_in_ms < tend)]#.reset_index(drop=True)
            frames_df['id_scene'] = id_scene
        else:
            frames_df = all_frames_df

        # I think you just need to get the same path, as the names of the images are defined wrt their frame in the video 
        # cur_scene_name = ".".join(video_path.split("/")[-1].split(".")[:-1])+f'-Scene{idx:03d}'
        # temp_dir_scene = os.path.join(temp_dir, cur_scene_name)
        temp_dir_scene = os.path.join(temp_dir, cur_video_name)

        aligned_frames_path_list, headpose_list, landmarks_3d_list, is_zoom_list = get_aligned_video_frames(frames_df, temp_dir=temp_dir_scene, verbose=verbose)
        
        # frames_df["aligned_frame_path"] = aligned_frames_path_list
        frames_df = frames_df.drop("path_to_frame", axis=1)
        frames_df["headpose"] = headpose_list
        frames_df["landmarks_3d"] = landmarks_3d_list
        frames_df['is_zoom'] = is_zoom_list

        # dropping the scene if there is a missing face; isnt it too harsh? 
        df_to_drop = frames_df.headpose.isnull()

        # if at least one to drop
        if df_to_drop.mean()>0:
            if df_to_drop.mean() == 1: 
                if verbose: print(f'No face detected in {100*df_to_drop.mean():.1f}% of the frames ... Cutting the scene {id_scene}')
                continue
            else:
                frames_df_ini = frames_df.drop('headpose', axis='columns').drop('landmarks_3d', axis='columns').copy()
                frames_df = frames_df[~df_to_drop]
                # new index used for the merge
                frames_df.reset_index(inplace=True)

                aligned_frames_path_list = [k for k in aligned_frames_path_list if k]

        frames_df = frames_df.join(pd.json_normalize(frames_df['headpose'])).drop('headpose', axis='columns')
        frames_df = frames_df.join(pd.json_normalize(frames_df['landmarks_3d'])).drop('landmarks_3d', axis='columns')

        detected_aus, au_intensities, facial_expression = [], [], []
        
        if model_choice == "joint_au_detection_intensity_estimator":
            detected_aus, au_intensities = get_au_intensities_and_detect_aus_video(aligned_frames_path_list, 
                                                                            device = device, 
                                                                            batch_size=batch_size,
                                                                            num_workers=num_workers,
                                                                            weights_download_dir=weights_download_dir)
        elif model_choice == "separate_prediction_heads":
            detected_aus = detect_action_units_video(aligned_frames_path_list, 
                                                    device=device,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    weights_download_dir=weights_download_dir)
            au_intensities = get_au_intensities_video(aligned_frames_path_list, 
                                                    device=device,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    weights_download_dir=weights_download_dir)
        else:
            print(f"Undefined model_choice = {model_choice} for get_facial_attributes_video()")
            raise NotImplementedError
        
        facial_expression = get_facial_expression_video(aligned_frames_path_list, 
                                                        device = device, 
                                                        batch_size=batch_size,
                                                        weights_download_dir=weights_download_dir)
        

        frames_df = frames_df.join(detected_aus)
        frames_df = frames_df.join(au_intensities)
        frames_df = frames_df.join(facial_expression)

        # fill the frames where no face detected with nans
        if df_to_drop.mean()>0:
            # put back the original index
            frames_df.set_index('index', inplace=True, drop=True)
            frames_df_processed = pd.DataFrame(np.nan, index=frames_df_ini.index, columns=frames_df.columns)
            frames_df_processed.loc[~df_to_drop, frames_df.columns] = frames_df.values
            frames_df_processed.loc[df_to_drop, frames_df_ini.columns] = frames_df_ini[df_to_drop].values
        else:
            frames_df_processed = frames_df

        list_frames_df_processed.append(frames_df_processed)
    
    # If several scenes
    if path_scene_cuts and (len(list_frames_df_processed)>1):
        frames_df_processed = pd.concat(list_frames_df_processed) 
    else: 
        try:
            list_frames_df_processed[0]
        except:
            print('There might be an error with the scenes cut file...')
            import sys
            sys.exit()

    if remove_temp:
        temp_dir = os.path.join(temp_dir, cur_video_name)
        print('Remove temp dir', temp_dir)
        shutil.rmtree(temp_dir)
        for dir_path in glob.glob(temp_dir+"-Scene*"):
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
    return frames_df_processed

def save_facial_attributes_video(video_path, 
                            output_save_path = "video_results.csv", 
                            model_choice:str="joint_au_detection_intensity_estimator",
                            temp_dir="./tmp", 
                            device="cpu",
                            batch_size = 256,
                            num_workers = 2,
                            weights_download_dir:str = "./weights_libreface", 
                            frame_interval=1,
                            remove_temp=False,
                            path_scene_cuts="/data/vbarrier/standup/videos/scenes/fr.csv"):
    frames_df = get_facial_attributes_video(video_path,
                                            model_choice=model_choice,
                                            temp_dir=temp_dir,
                                            device=device, 
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            weights_download_dir=weights_download_dir,
                                            frame_interval=frame_interval,
                                            remove_temp=remove_temp,
                                            path_scene_cuts=path_scene_cuts)
    save_path = uniquify_file(output_save_path)
    frames_df.to_csv(save_path, index=False)
    print(f"Facial attributes of the video saved to {save_path}")
    return save_path

def save_facial_attributes_image(image_path, 
                                 output_save_path = "image_results.csv", 
                                 model_choice:str="joint_au_detection_intensity_estimator",
                                 temp_dir="./tmp", 
                                 device="cpu",
                                 weights_download_dir:str = "./weights_libreface"):
    attr_dict = get_facial_attributes_image(image_path, 
                                            model_choice=model_choice, 
                                            temp_dir=temp_dir, 
                                            device=device,
                                            weights_download_dir=weights_download_dir)
    for k, v in attr_dict.items():
        attr_dict[k] = [v]
    attr_df = pd.DataFrame(attr_dict)
    attr_df = attr_df.join(pd.json_normalize(attr_df['detected_aus'])).drop('detected_aus', axis='columns')
    attr_df = attr_df.join(pd.json_normalize(attr_df['au_intensities'])).drop('au_intensities', axis='columns')
    attr_df.index.name = 'frame_idx'
    save_path = uniquify_file(output_save_path)
    attr_df.to_csv(save_path, index=False)
    print(f"Facial attributes of the image saved to {save_path}")
    return save_path

def get_facial_attributes(file_path, 
                          output_save_path=None, 
                          model_choice:str="joint_au_detection_intensity_estimator",
                          temp_dir="./tmp", 
                          device="cpu",
                          batch_size = 256,
                          num_workers = 2,
                          weights_download_dir:str = "./weights_libreface",
                          frame_interval=1,
                          remove_temp=False,
                          path_scene_cuts=""):
    file_type = check_file_type(file_path)
    if file_type == "image":
        if output_save_path is None:
            return get_facial_attributes_image(file_path, model_choice=model_choice, 
                                               temp_dir=temp_dir, device=device, weights_download_dir=weights_download_dir)
        else:
            try:
                return save_facial_attributes_image(file_path, output_save_path=output_save_path, 
                                                    model_choice=model_choice, temp_dir=temp_dir, 
                                                    device=device, weights_download_dir=weights_download_dir)
            except Exception as e:
                print(e)
                print("Some error in saving the results.")
    elif file_type == "video":
        if output_save_path is None:
            return get_facial_attributes_video(file_path, model_choice=model_choice, 
                                               temp_dir=temp_dir, device=device, 
                                               batch_size=batch_size, 
                                               num_workers=num_workers, weights_download_dir=weights_download_dir, frame_interval=frame_interval,
                                               remove_temp=remove_temp,path_scene_cuts=path_scene_cuts)
        else:
            try:
                return save_facial_attributes_video(file_path, output_save_path=output_save_path, 
                                                    model_choice=model_choice, temp_dir=temp_dir, 
                                                    device=device, batch_size=batch_size, 
                                                    num_workers=num_workers, weights_download_dir=weights_download_dir, frame_interval=frame_interval,
                                                    remove_temp=remove_temp,path_scene_cuts=path_scene_cuts)
            except Exception as e:
                print(e)
                print("Some error in saving the results.")
                return False
