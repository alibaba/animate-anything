"""
Used to compress video in: https://github.com/ArrowLuo/CLIP4Clip
Author: ArrowLuo
"""
import os
import argparse
import ffmpeg
import subprocess
import time
import multiprocessing
from multiprocessing import Pool
import shutil
import json
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
# multiprocessing.freeze_support()

def compress(paras):
    input_video_path, output_video_path = paras
    try:
        command = ['ffmpeg',
                   '-y',  # (optional) overwrite output file if it exists
                   '-i', input_video_path,
                   '-filter:v',
                   'scale=\'if(gt(a,1),trunc(oh*a/2)*2,512)\':\'if(gt(a,1),512,trunc(ow*a/2)*2)\'',  # scale to 256
                   '-map', '0:v',
                   #'-r', '3',  # frames per second
                   output_video_path,
                   ]
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        retcode = ffmpeg.poll()
        # print something above for debug
    except Exception as e:
        raise e

def prepare_input_output_pairs(input_root, output_root):
    input_video_path_list = []
    output_video_path_list = []
    for root, dirs, files in os.walk(input_root):
        for file_name in files:
            input_video_path = os.path.join(root, file_name)
            output_video_path = os.path.join(output_root, file_name)
            output_video_path = os.path.splitext(output_video_path)[0] + ".mp4"
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                pass
            else:
                input_video_path_list.append(input_video_path)
                output_video_path_list.append(output_video_path)
    return input_video_path_list, output_video_path_list

def msvd():
    captions = pickle.load(open('raw-captions.pkl','rb'))
    outdir = "/data/datasets/msvd/videos_mp4"
    for key in captions:
        outpath = os.path.join(outdir, key+".txt")
        with open(outpath, 'w') as f:
            for line in captions[key]:
                f.write(" ".join(line)+"\n")

def webvid():

    df = pd.read_csv('/webvid/results_2M_train_1/0.csv')
    df['rel_fn'] = df.apply(lambda x: os.path.join(str(x['page_dir']), str(x['videoid'])), axis=1)

    df['rel_fn'] = df['rel_fn'] + '.mp4'
        # remove nan
    df.dropna(subset=['page_dir'], inplace=True)

    playlists_to_dl = np.sort(df['page_dir'].unique())

    vjson = []
    video_dir = '/webvid/webvid/data/videos'
    for page_dir in playlists_to_dl:
        vid_dir_t = os.path.join(video_dir, page_dir)
        pdf = df[df['page_dir'] == page_dir]
        if len(pdf) > 0:
            for idx, row in pdf.iterrows():
                video_fp = os.path.join(vid_dir_t, str(row['videoid']) + '.mp4')
                if os.path.isfile(video_fp):
                    caption = row['name']
                    video_path = os.path.join(page_dir, str(row['videoid'])+'.mp4')
                    vjson.append({'caption':caption,'video':video_path})
    with open('/webvid/webvid/data/2M.json', 'w') as f:
        json.dump(vjson, f)

def webvid20k():
    j = json.load(open('/webvid/webvid/data/2M.json'))
    idir = '/webvid/webvid/data/videos'

    v2c = []
    for item in j:
        caption = item['caption']
        video = item['video']
        if os.path.exists(os.path.join(idir, video)):
            v2c.append(item)
    print("video numbers", len(v2c))
    with open('/webvid/webvid/data/40K.json', 'w') as f:
        json.dump(v2c, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress video for speed-up')
    parser.add_argument('--input_root', type=str, help='input root')
    parser.add_argument('--output_root', type=str, help='output root')
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root

    assert input_root != output_root

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    input_video_path_list, output_video_path_list = prepare_input_output_pairs(input_root, output_root)

    print("Total video need to process: {}".format(len(input_video_path_list)))
    num_works = cpu_count()
    print("Begin with {}-core logical processor.".format(num_works))

    pool = Pool(num_works)
    data_dict_list = pool.map(compress,
                              [(input_video_path, output_video_path) for
                               input_video_path, output_video_path in
                               zip(input_video_path_list, output_video_path_list)])
    pool.close()
    pool.join()

    print("Compress finished, wait for checking files...")
    for input_video_path, output_video_path in zip(input_video_path_list, output_video_path_list):
        if os.path.exists(input_video_path):
            if os.path.exists(output_video_path) is False or os.path.getsize(output_video_path) < 1.:
                print("convert fail: {}".format(output_video_path))