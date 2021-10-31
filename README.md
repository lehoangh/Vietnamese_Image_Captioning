0. Installation
- Nên cài python environment (conda or virtualenv) và cài các gói cần thiết trong file requirements.txt như sau:
    pip install -r requirements.txt

1. Chạy python download_imgs.py (xem chi tiết ở file .py) để download ảnh về máy
2. Chạy python build_vocab.py để xây dựng bộ vocabulary của từ tiếng việt
3. Chạy python resize.py để resize ảnh về đúng format dùng trong bộ ResNet125 encoder
4. Chạy python train.py để huấn luyện 
5. (Optinal) Chạy python resume_training.py để huấn luyện tiếp khi bị ngắt/stop vì lý do gì đó
6. Chạy test thử bằng python sample.py --image tên_file_ảnh_muốn_test_trong_thư_mục_images_to_sample

