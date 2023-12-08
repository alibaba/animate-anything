""""
调用pySceneDetect库进行视频场景切分，过滤掉视频长度小于3s的clips
https://github.com/Breakthrough/PySceneDetect
conda activate XPretrain
"""
import os
import argparse
import logging
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, ContentDetector
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl


# file to record processed videos
logfile_path = "./log/tmp_processed_videos_adaptive.txt"

def process_video(video_path):
    # start/end times of all scenes found in the video
    print("detect scene", video_path)
    scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=0.1)) # 1.0
    # scene_list = detect(video_path, ContentDetector())
    # filter clips <= 3.0s
    scene_list = list(filter(lambda scene : 60.0 >=(scene[1].get_seconds()-scene[0].get_seconds()) >= 2.0, scene_list))
    num_scene = len(scene_list)

    logging.info('------scene_list-------')
    logging.info(len(scene_list))

    # save clips into current dir
    split_video_ffmpeg(video_path, scene_list, show_progress=True)



def process_and_log(video_path):
    process_video(video_path)
    with open(logfile_path, 'a') as logfile:
        logfile.write(os.path.basename(video_path) + '\n')
        logfile.flush()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def fun(folder_path, vid_format='mp4'):
    logging.info(f'supported video format: {vid_format}')

    file_list = os.listdir(folder_path)
    file_list = list(filter(lambda fn: fn.endswith(f'.{vid_format}'), file_list))
    
    logging.info(f'total videos: {len(file_list)}')
    logging.info('output videos will be saved into current dir.')

    # 读取已处理过的文件列表
    processed_files = set()
    if os.path.exists(logfile_path):
        with open(logfile_path, 'r') as logfile:
            processed_files = set(logfile.read().splitlines())

    # processed_files = ()
    # lock = threading.Lock()


    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []

        for i, file_name in enumerate(file_list):
            if(i % 100 == 0):
                logging.info(f'done processing {i} videos')
            if(file_name in processed_files):
                logging.info(f'{file_name} processed before, ignored...')
                continue
            if file_name.endswith(f".{vid_format}"):
                video_path = os.path.join(folder_path, file_name)
                logging.info(f'processing video {file_name}')
                # 提交视频处理任务给线程池，并将Future对象存储到列表中
                future = executor.submit(process_and_log, video_path)
                futures.append(future)

                # 避免短时间提交过多任务导致内存不足
                if len(futures) >= 24:
                    # 等待最早完成的任务完成
                    for completed_future in as_completed(futures):
                        completed_future.result()
                    # 清空已完成的任务列表
                    futures.clear()

        # 等待剩余的任务完成
        for completed_future in as_completed(futures):
            completed_future.result()

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Process a folder path.')

    parser.add_argument('folder_path', type=str, help='Path to the folder to be processed', default='/data_menghao/XPretrain/hdvila_100m/default_content_detector_clips_the_magic_key_from_bbdown')

    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        logging.info(f"Error: The folder {args.folder_path} does not exist.")
        return
    
    
    fun(folder_path=args.folder_path, vid_format='mp4')
    fun(folder_path=args.folder_path, vid_format='mkv')
    fun(folder_path=args.folder_path, vid_format='avi')
    

if __name__ == '__main__':
    main()