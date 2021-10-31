import os
import requests
import json
import argparse

"""Download ảnh từ đường link 'coco_url'(e.g.  cocodataset.org) trong file json"""

import sys
import os
import requests
import time
import getopt
import concurrent.futures


def show_help():
    """Show help."""
    print(f"""Usage: {sys.argv[0]} [OPTIONS]
  -j, --json_file   JSON file contains urls of images to download
  -h, --help        Show this help
  -d, --destination Folder to save the files (Default current dir)
  -t, --threads     Number of concurrent downloads (Default ThreadPoolExecutor default)
  -n, --number      Number of images to download (Default 20)

  Example:
  python {sys.argv[0]} -j ./data/annotations/uitviic_captions_train2017.json -d ~/Pictures -t5 -n10

    """)
    sys.exit(0)


def download_image(img_url, destination):
    """Function to download image."""
    title = img_url.split('/')[-1]
    print("File name: ", title)
    print("Downloading file: %s " %title)

    time.sleep(0.1)  # for the next "Start" message show up after the previous "Downloaded"
    print(f'Started download of {title}')

    img_blob = requests.get(img_url, timeout=5, stream=True).content
    if not os.path.exists(destination):
        os.mkdir(destination)
    with open(destination + '/' + title, 'wb') as img_file:
        img_file.write(img_blob)

    # print(data_train['images'])
    # Trích xuất url của bức ảnh


        # r = requests.get(img_url, stream=True)
        # print(r)
        # folder_name = label_set + "/"
        # print(folder_name)
        # # Kiểm tra có thư mục chưa
        # if not os.path.isdir(folder_name):
        #     os.mkdir(folder_name)
        # file_path = folder_name + file_name
        # print(file_path)
        # # this should be file_name variable instead of "file_name" string
        # if not os.path.exists(file_path):
        #     with open(file_path, 'wb') as f:
        #         for chunk in r:
        #             f.write(chunk)

    return title


def main(args, opts):
    if len(sys.argv) == 1:
        print('Enter the keywords')
        show_help()

    threads = None
    destination = os.path.realpath(os.curdir)
    number = 20

    for o, v in opts:
        print(o, v)

        if o in ['-h', '--help']:
            show_help()
        elif o in ['-j', '--json_file']:
            json_file = v
        elif o in ['-d', '--destination']:
            destination = os.path.realpath(v)
        elif o in ['-t', '--threads']:
            threads = int(v)
        elif o in ['-n', '--number']:
            number = v
        else:
            raise AssertionError('Unhandled option')

    start = time.perf_counter()

    # Mở file chứa các dữ liệu ảnh cần download
    with open(json_file) as f:
        data_train = json.load(f)
        print(data_train.keys())

    # Download các ảnh vào thư mục
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:

        results = [executor.submit(download_image, img_info['coco_url'], destination) for img_info in data_train['images']]
        for t in concurrent.futures.as_completed(results):
            print(f'Downloaded {t.result()}')

    stop = time.perf_counter()

    print(f'Finished in {stop-start} seconds')


# # Đọc file các caption
# def load_doc(filename):
# 	# open the file as read only
# 	file = open(filename, 'r')
# 	# read all text
# 	text = file.read()
# 	# close the file
# 	file.close()
# 	return text

# def read_json_file(label_set, json_file):

        
#         # return data_train

if __name__ == "__main__":
    # json_file = "data/uitviic_captions_train2017.json"
    # read_json_file("data/train", json_file)
    # download_image()
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('-h', '--help', type=str,
    # #                     help="Show help")
    # parser.add_argument('-j', '--json_file', type=str, default='./data/annotations/uitviic_captions_train2017.json',
    #                     help='JSON file contains url links of images')
    # parser.add_argument('-d', '--destination', type=str, default='./data/train2017/',
    #                     help='Folder to save the files')
    # parser.add_argument('-t', '--threads', type=int,
    #                     help='Number of concurrent downloads (Default ThreadPoolExecutor default)')
    # parser.add_argument('-n', '--number', type=int, default=20,
    #                     help='Number of images to download (Default 20)')
    # args = parser.parse_args()
    # print(args)
    defopts = ['help', 'json_file=', 'destination=', 'threads=', 'number=']
    # json_file = args.json_file
    # print(json_file)
    opts, args = getopt.getopt(sys.argv[1:], 'hj:d:t:n:', defopts)
    print(opts, args)
    main(args, opts)