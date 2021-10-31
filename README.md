https://github.com/sauravraghuvanshi/Udacity-Computer-Vision-Nanodegree-Program/tree/master/project_2_image_captioning_project  
and: https://medium.com/@deepeshrishu09/automatic-image-captioning-with-pytorch-cf576c98d319  

0. Installation
- Nên cài python environment (conda or virtualenv) và cài các gói cần thiết trong file requirements.txt như sau:
    pip install -r requirements.txt

1. Chạy python download_imgs.py (xem chi tiết ở file .py) để download ảnh về máy
2. Chạy python build_vocab.py để xây dựng bộ vocabulary của từ tiếng việt
3. Chạy python resize.py để resize ảnh về đúng format dùng trong bộ ResNet125 encoder
4. Chạy python train.py để huấn luyện 
5. (Optinal) Chạy python resume_training.py để huấn luyện tiếp khi bị ngắt/stop vì lý do gì đó
6. Chạy test thử bằng python sample.py --image tên_file_ảnh_muốn_test_trong_thư_mục_images_to_sample

Chưa có bước evaluation, nhưng nhận thấy mô hình (ResNet125 + LSTM) bị vướng vào loss 0.4x và 0.5x
...
Epoch [10/20], Step [0/106], Loss: 0.4887, Perplexity: 1.6301               
Epoch [10/20], Step [10/106], Loss: 0.5264, Perplexity: 1.6929  
Epoch [10/20], Step [20/106], Loss: 0.5330, Perplexity: 1.7040  
Epoch [10/20], Step [30/106], Loss: 0.5314, Perplexity: 1.7014  
Epoch [10/20], Step [40/106], Loss: 0.5335, Perplexity: 1.7049  
Epoch [10/20], Step [50/106], Loss: 0.5307, Perplexity: 1.7000  
Epoch [10/20], Step [60/106], Loss: 0.5492, Perplexity: 1.7319  
Epoch [10/20], Step [70/106], Loss: 0.5334, Perplexity: 1.7046  
Epoch [10/20], Step [80/106], Loss: 0.5661, Perplexity: 1.7614  
Epoch [10/20], Step [90/106], Loss: 0.5931, Perplexity: 1.8095  
Epoch [10/20], Step [100/106], Loss: 0.5456, Perplexity: 1.7257  
Epoch [11/20], Step [0/106], Loss: 0.4941, Perplexity: 1.6390  
Epoch [11/20], Step [10/106], Loss: 0.5443, Perplexity: 1.7234  
Epoch [11/20], Step [20/106], Loss: 0.5243, Perplexity: 1.6893  
Epoch [11/20], Step [30/106], Loss: 0.5362, Perplexity: 1.7094  
Epoch [11/20], Step [40/106], Loss: 0.4876, Perplexity: 1.6283  
Epoch [11/20], Step [50/106], Loss: 0.5179, Perplexity: 1.6785  
Epoch [11/20], Step [60/106], Loss: 0.5540, Perplexity: 1.7402  
Epoch [11/20], Step [70/106], Loss: 0.5082, Perplexity: 1.6623  
Epoch [11/20], Step [80/106], Loss: 0.5183, Perplexity: 1.6791  
Epoch [11/20], Step [90/106], Loss: 0.5311, Perplexity: 1.7009  

nên mình đã stop không train nữa 

Sau đó, mình thử test mô hình `encoder-11-1-20211031_153236.ckpt` và mô hình `decoder-11-1-20211031_153236.ckpt` (số `11` trong tên file tức là epoch thứ 10) với những ảnh trong folder `images_to_sample` thì nhận thấy mô hình không phân biệt được những môn thể thao có bóng và những môn thể thao KHÔNG có bóng  
Với bức ảnh `hai_co_gai`, cũng chỉ nhận diện được một cô với caption: `Người phụ_nữ đang cầm bóng chống hông chụp hình .`  
Với bức ảnh `bong_da_tre_em`, sinh mô tả được với `Những cầu_thủ bóng_đá trẻ đang thi_đấu trên sân .`  
Với bức ảnh `bong_da_nguoi_lon`, sinh mô tả được với `Các cầu_thủ bóng_đá đang thi_đấu ở trên sân .` nhưng không hẳn là thi đấu mà đứng ngoài sân nhỉ (?)  
Với bức ảnh `co_gai_hockey`, sinh mô tả `Một nữ vận_động_viên tennis đang chuẩn_bị đánh_bóng .`  
Với bức ảnh `pickleball`, sinh mô tả `Người đàn_ông đang cầm vợt tennis đỡ bóng thấp .`  

Mọi người có thể copy các ảnh vào thư mục `images_to_sample` rồi test thử với bước số 6 ở trên và chọn mô hình (checkpoints) muốn dùng ở file `config.py` với `encoder_path` và `decoder_path`.

Models đã trained: https://drive.google.com/drive/folders/1-GfaseydLVCidHuM8qttxD_uYUydwzpK?usp=sharing
