import os, shutil, cv2, math, random
import time, yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from craft import CRAFT
import file_utils, craft_utils, imgproc
import warnings

# 경고 메시지 감추기
warnings.simplefilter("ignore")


def get_config(file, verbose=False):
    f = open(file)
    config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def resize_and_save_images(src_dir, dst_dir):
    # 출력 디렉토리가 존재하면 삭제 후 새로 생성
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    img_files, _, _ = file_utils.get_files(src_dir)
    if len(img_files) == 0:
        raise AssertionError(f"입력파일이 없습니다.- dir: {src_dir} # of img: {len(img_files)}")

    for img_file in img_files:
        img = cv2.imread(img_file)
        h, w = img.shape[:2]

        if max(h, w) > 500:
            if h > w:
                new_h = 500
                new_w = int((500 / h) * w)
            else:
                new_w = 500
                new_h = int((500 / w) * h)

            resized_img = cv2.resize(img, (new_w, new_h))
            cv2.imwrite(os.path.join(dst_dir, os.path.basename(img_file)), resized_img)
        else:
            shutil.copy(img_file, dst_dir)

    print(f"Images processed and saved in {dst_dir}")


def get_char_bounding_boxes(textmap, text_threshold, low_text):
    # textmap을 복사합니다.
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    # 텍스트 점수 맵에 대해 thresholding을 수행합니다.
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)

    # 연결된 구성 요소를 찾습니다.
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8), connectivity=4)

    det = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # segmentation map을 만듭니다.
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        # 경계 확인
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # contour를 찾고 최소 사각형을 만듭니다.
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # 다이아몬드 모양 정렬
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # 시계 방향 정렬
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)

    return np.array(det)



def draw_bounding_boxes(image, boxes, save_fig=False, save_imgfname="test.png"):
    """
    원본 이미지에 bounding box를 그린 후 시각화하고 저장 옵션이 켜져 있는 경우 이미지를 저장합니다.

    Parameters:
    image (numpy.ndarray): 원본 이미지 배열 (H, W)
    boxes (numpy.ndarray): bounding box 좌표 배열 (num_boxes, 4, 2)
    save_fig (bool): 이미지를 파일로 저장할지 여부 (기본값: False)
    save_imgfname (str): 저장할 파일 이름 (기본값: "test.png")

    Returns:
    None
    """
    # 이미지 복사 및 색상 채널 확인
    if len(image.shape) == 2:  # Grayscale 이미지인 경우
        image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:  # 컬러 이미지인 경우
        image_copy = image.copy()
    
    # 색상 리스트 생성
    colors = list(mcolors.CSS4_COLORS.values())
    random.shuffle(colors)  # 색상 리스트를 무작위로 섞기

    # 각 bounding box를 그립니다.
    for i, box in enumerate(boxes):
        box = np.int0(box)  # 정수형으로 변환
        color = colors[i % len(colors)]  # 무작위 색상 선택
        color_rgb = mcolors.hex2color(color)
        color_rgb = [int(c * 255) for c in color_rgb]  # OpenCV 색상 형식으로 변환
        cv2.polylines(image_copy, [box], isClosed=True, color=color_rgb, thickness=2)

    # matplotlib을 사용하여 이미지 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image_copy)
    plt.axis('off')  # 축을 숨깁니다.

    # 이미지를 저장하는 옵션
    if save_fig:
        plt.savefig(save_imgfname, bbox_inches='tight', pad_inches=0)
    
    plt.show()


def sort_boxes_by_line(boxes, line_threshold=10):
    """
    글자의 bounding box를 줄 단위로 정렬합니다.

    Parameters:
    boxes (numpy.ndarray): bounding box 좌표 배열 (num_boxes, 4, 2)
    line_threshold (int): 줄을 구분하는 임계값

    Returns:
    sorted_boxes (list): 줄 단위로 정렬된 bounding box 리스트
    """
    # 상단 y 좌표를 기준으로 정렬
    boxes = sorted(boxes, key=lambda box: np.min(box[:, 1]))

    # 줄 단위로 나누기
    lines = []
    current_line = []
    current_y_min = np.min(boxes[0][:, 1])
    current_y_max = np.max(boxes[0][:, 1])

    for box in boxes:
        y_min = np.min(box[:, 1])
        y_max = np.max(box[:, 1])

        if y_min > current_y_max + line_threshold:
            lines.append(current_line)
            current_line = [box]
            current_y_min = y_min
            current_y_max = y_max
        else:
            current_line.append(box)
            current_y_min = min(current_y_min, y_min)
            current_y_max = max(current_y_max, y_max)
    
    if current_line:
        lines.append(current_line)

    # 각 줄을 왼쪽에서 오른쪽으로 정렬
    sorted_boxes = []
    for line in lines:
        sorted_boxes.extend(sorted(line, key=lambda box: np.min(box[:, 0])))

    return sorted_boxes


def save_cropped_letters(image_fpath, boxes, out_dir, line_threshold=10):
    """
    원본 이미지에서 bounding box를 이용하여 각 글자 이미지를 잘라 저장합니다.

    Parameters:
    image (numpy.ndarray): 원본 이미지 배열 (H, W) 또는 (H, W, 3)
    boxes (numpy.ndarray): bounding box 좌표 배열 (num_boxes, 4, 2)
    out_dir (str): 출력 디렉토리 경로
    line_threshold (int): 줄을 구분하는 임계값

    Returns:
    None
    """
    image = imgproc.loadImage(image_fpath)
    filename, file_ext = os.path.splitext(os.path.basename(image_fpath))
    
    # 출력 디렉토리가 존재하면 삭제 후 새로 생성
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 이미지가 흑백인지 컬러인지 확인
    is_color = len(image.shape) == 3

    # 줄 단위로 글자 정렬
    sorted_boxes = sort_boxes_by_line(boxes, line_threshold)

    for i, box in enumerate(sorted_boxes):
        # bounding box 좌표를 이용하여 이미지 잘라내기
        box = np.int0(box)
        x_min, x_max = np.min(box[:, 0]), np.max(box[:, 0])
        y_min, y_max = np.min(box[:, 1]), np.max(box[:, 1])
        
        cropped_image = image[y_min:y_max, x_min:x_max]

        # 파일 저장 경로
        out_path = os.path.join(out_dir, f"{filename}_out_{i}.png")
        
        # 이미지 저장 (컬러인 경우 BGR로 변환하여 저장)
        if is_color:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, cropped_image)

    print(f"Cropped letters saved in {out_dir}")





def main(args):
    ## [PARAMS SETTING] ##
    config = get_config(args.config, verbose=False)
    model_path = config["model_path"]
    use_cuda = True
    refine = False
    img_folder = config["img_folder"]
    result_folder = config["result_folder"]

    ## Default option
    text_threshold = config["text_threshold"]
    low_text = config["low_text"]
    link_threshold = 0.4         # link confidence threshold
    canvas_size = 1280           # image size for inference
    mag_ratio = 1.5              # image magnification ratio
    poly = False                 # enable polygon type
    show_time = True
    refiner_model = ""
    verbose = str2bool(config["verbose"])


    ## [Resizing raw imgs]
    tmp_dir = "../resized_imgs"
    resize_and_save_images(img_folder, tmp_dir)
    test_folder = tmp_dir

    ## [Read img & load network]
    image_list, _, _ = file_utils.get_files(test_folder)
    print(f"test folder내 이미지수: {len(image_list)}")

    # 디렉토리 존재 시 덮어씀
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.makedirs(result_folder)


    net = CRAFT()     # initialize
    if use_cuda:
            net.load_state_dict(copyStateDict(torch.load(model_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # 미세 조정 network
    refine_net = None
    if refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        poly = True


    ## [Inference]
    t = time.time()

    cuda = use_cuda
    for k, img_path in enumerate(image_list):

        image = imgproc.loadImage(img_path)
        print(f"input img ({os.path.basename(img_path)}) & shape: {image.shape}")

        # resizing 후 이미지의 비율(bnd box position 복구 용) 저장
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        
        if verbose:
            print(f"canvas size: {canvas_size}")                  # 1280
            print(f"img resized: {img_resized.shape}")            # (H, W, 3) (160, 480, 3)   -  resized된 이미지
            print(f"ratio: {target_ratio, ratio_h, ratio_w}")     # (1.5, 0.6666666666666666, 0.6666666666666666)
            print(f"size heatmap: {size_heatmap}\n")              # (240, 80)
        
        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)           
        x = torch.from_numpy(x).permute(2, 0, 1)                  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                              # [c, h, w] to [b, c, h, w]  (1, 3, 160, 480)
        if cuda:
            x = x.cuda()
        
        # forward pass
        with torch.no_grad():
            y, feature = net(x)                                   # y는 main out, feature는 y를 보정하는데 쓰이는 extra feature
        
        if verbose:
            print(f"y, feature shape: {y.shape}, {feature.shape}")    # ([1, 80, 240, 2]), ([1, 32, 80, 240])
        
        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()                # Character Region Map (문자 영역 맵)의 초기 ver.
        score_link = y[0,:,:,1].cpu().data.numpy()                # Character Affinity Map (문자 연결 맵)의 초기 ver.
        if verbose:
            print(f"score_text, score_link shape: {score_text.shape}, {score_link.shape}")    # (80, 240), (80, 240)
        
        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy
        
        
        ## Activation map result 저장
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img) 
        
        if verbose:
            print(f"render_img shape: {render_img.shape}")               # (80, 480)
            print(f"Activate map img shape: {ret_score_text.shape}")     # (80, 480, 3)
        
        filename, file_ext = os.path.splitext(os.path.basename(img_path))
        mask_fpath = os.path.join(result_folder, f"{filename}_actv.jpg")
        cv2.imwrite(mask_fpath, ret_score_text)
        print(f"score_text shape: {score_text.shape}")                    # (80, 480, 3)
        
        ## character bnd box result 저장
        boxes = get_char_bounding_boxes(score_text, text_threshold, low_text)
        new_boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        
        if verbose:
            print(f"box threshold - text_threshold:{text_threshold}, low_text: {low_text}")
            print(f"boxes shape: {boxes.shape}")  # 출력: (num_boxes, 4, 2)
        
        # preview plot
        fig_fpath = os.path.join(result_folder, f"{filename}_preview.jpg")
        draw_bounding_boxes(image, new_boxes, save_fig=True, save_imgfname=fig_fpath)
        
        
        save_cropped_letters(img_path, new_boxes, result_folder)

    if show_time:
        print("elapsed time : {}s".format(time.time() - t))



def str2bool(v):
    if isinstance(v, bool):
        return v
    elif isinstance(v, int):
        return v
    
    if v.lower() in ('true', 't', '1', 'True', 'TRUE'):
        return True
    elif v.lower() in ('false', 'f', '0', 'False', 'FALSE'):
        return False
    elif v.lower() in ('None', 'none', ""):
        return None
    else:
        raise AssertionError("check boolean value error:", v)
        


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args)