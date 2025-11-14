# train_taco_yolov8n_fix.py
import os
from ultralytics import YOLO

data_yaml = "/home/linux/yolov8_ws/dataset/open_images_dataset/data.yaml"
model_name = "yolov8n.pt"
# model_name = "/home/linux/yolov8_ws/runs/open_dataset_test1/weights/last.pt"
project = "runs"
run_name = "open_dataset_test2"

model = YOLO(model_name)
# model.load('yolov8n.pt')

results = model.train(
    # 1. 모델 및 데이터
    data=data_yaml,     # type: str, default: null, 데이터셋 yaml 파일 경로
    imgsz=768,          # type: int, default: 640
    # task='detect',    # type: str, (detect, segment, classify)
    # classes=[],       # type: list, default: 모든 클래스, 학습할 클래스 인덱스 리스트
    # rect=False,       # type: bool, default: False, 이미지 크기 비율 유지
    # cache=False,      # type: bool, default: False, 데이터셋 캐시 사용

    # 2. 학습 제어 파라미터
    epochs=200,             # type: int, default: 100, 학습 에폭 수
    batch=16,               # type: int, default: 16, 배치 크기
    device=0,               # type: int/str, default: '', 학습에 사용할 장치 (예: '0' 또는 'cuda:0')
    resume=True,         # type: bool/str, default: False, 중단된 학습 재개 또는 특정 체크포인트에서 재개
    # exist_ok=False,       # type: bool, default: False, 기존 실험 디렉토리 덮어쓰기 허용 여부
    project=project,        # type: str, default: runs/train, 결과 저장 기본 폴더 명
    name=run_name,          # type: str, default: exp, 실험 이름

    # 3. 최적화 관련
    # optimizer="AdamW",    # type: str, default: SGD, 옵티마이저 종류 (SGD, Adam, AdamW 등)
    # lr0=0.01,             # type: float, default: 0.01, 초기 학습률
    # lrf=0.01,             # type: float, default: 0.01, 마지막 학습률 비율 (최종 학습률 = lrf * lr0)
    # momentum=0.937,       # type: float, default: 0.937, SGD 옵티마이저 모멘텀 값
    # weight_decay=0.0005,  # type: float, default: 0.0005, 가중치 감쇠
    warmup_epochs=3,      # type: int, default: 3, 학습률 워밍업 에폭 수
    # warmup_momentum=0.8,  # type: float, default: 0.8, 워밍업 시작시 모멘텀
    cos_lr=True,          # type: bool, default: True, 코사인 학습률 스케줄러 사용 여부

    # 4. 손실 및 학습 전략
    # box=7.5,              # type: float, default: 7.5, bbox 손실 가중치
    # cls=0.5,              # type: float, default: 0.5, class 손실 가중치
    # dfl=1.5,              # type: float, default: 1.5, distribution focal loss 가중치
    # label_smoothing=0.0,  # type: float, default: 0.0, 라벨 스무딩 (0 ~ 1 사이 값)
    # anchor_auto=True,     # type: bool, default: True, anchor 자동 조정 여부 (yolov8은 anchor-free 구조이므로 무시될 수 있음)
    # nbs=64,               # type: int, default: 64, nominal batch size (normalize batch size 계산용)
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,    # type: float, default: 0.015, 0.7, 0.4, 색상, 채도, 명도 증강 비율
    degrees=5.0, translate=0.15, scale=0.7, shear=2.0,     # type: float, default: 0.0, 0.1, 0.5, 0.0, 데이터 증강 파라미터(회전/이동/스케일/왜곡)
    flipud=0.0, fliplr=0.5,   # type: float, default: 0.0, 0.5, 상하/좌우 반전 확률
    mosaic=0.6,           # type: float, default: 1.0, 4장을 합성해 여러 스케일/배경/배치를 한 번에 보게함
    mixup=0.05,            # type: float, default: 0.0, 믹스업 증강 비율
    # copy_paste=0.05,      # type: float, default: 0.0, 복/붙 증강 비율

    # 5. 검증 및 로깅 관련
    # val=True,             # type: bool, default: True, 에폭마다 validation 수행 여부
    # save=True,            # type: bool, default: True, 최종 모델 weight 저장 여부
    # save_period=-1,       # type: int, default: -1, 몇 epoch마다 저장할지 (-1: 마지막만 저장)
    # plots=True,           # type: bool, default: True, 시각화 플롯 저장 여부
    # verbose=False,        # type: bool, default: False, 세부 학습 로그 출력 여부
    # seed=0,               # type: int, default: 0, 재현성을 위한 random seed
    # deterministic=False,  # type: bool, default: False, 완전 재현성 모드 (느려짐)

    # 6. 분산 및 고급 설정
    # sync_bn=False,        # type: bool, default: False, 분산 학습 시 batch normalization 동기화
    workers=2,            # type: int, default: 8, DataLoader 병렬 처리 개수
    close_mosaic=20,      # type: int, default: 10, 마지막 몇 epoch 동안 모자이크 증강 비활성화
    # overlap_mask=True,    # type: bool, default: True, segmentation 시 마스크 중첩 허용 여부
    # mask_ratio=4.0,       # type: float, default: 4.0, segmentation 시 마스크 손실 비율
    # dropout=0.0,          # type: float, default: 0.0, 분류 모델 dropout 비율
)

best_ckpt = os.path.join(project, run_name, "weights", "best.pt")
metrics = model.val(
    # 1. 입력 설정
    data=data_yaml,         # type: str, 검증용 데이터셋 설정 파일 경로 (data.yaml)
    imgsz=768,             # type: int/list[int], default: 640, 입력 이미지 크기
    batch=16,               # type: int, default: 16, 검증 시 배치 크기
    device=0,               # type: str, 검증에 사용할 장치 (0, cpu 등)
    model=best_ckpt,        # type: str, 검증할 모델 파일 경로 (.pt, .yaml)
    # split='val',          # type: str, default: 'val', 데이터셋 분할 (train, val, test)
    # workers=8,            # type: int, default: 8, 데이터 로딩을 위한 CPU 쓰레드
    # rect=True,              # type: bool, default: False, 원본 이미지 비율 유지 여부 (rectangular val)
    # classes=list[int]/None,       # type: list, default: 모든 클래스, 특정 클래스만 검증하고 싶을 때 지정
    # max_batch_size=16,    # type: int, default: 16, 검증 시 최대 batch size 제한
    # seed=0,               # type: int, default: 0, 재현성을 위한 random seed

    # 2. 검출 조건
    # conf=0.001,             # type: float, default: 0.001, 최소 confidence 임계값 (탐지 확률 임계값)
    # iou=0.7,                # type: float, default: 0.6, 검증/추론 시 IoU 임계값
    # max_det=300,          # type: int, default: 300, 한 이미지당 최대 검출 객체 수
    # conf_thres=0.001,     # type: float, default: 0.001, confidence 임계값 (conf와 동일)
    # iou_thres=0.6,        # type: float, default: 0.6, IoU 임계값 (iou와 동일)
    # agnostic_nms=False,   # type: bool, default: False, 클래스 무시한 NMS 사용 여부
    # nms_time_limit=1.0,   # type: float, default: 1.0, NMS 수행 시간 제한 (초)

    # 3. 저장 옵션
    # save=True,            # type: bool, default: True, 검증 결과 이미지 저장 여부
    # save_json=False,      # type: bool, default: False, COCO 형식 결과(JSON) 저장 여부
    # save_json_path='runs/val/coco_eval.json',     # type: str, default: 'runs/val/coco_eval.json', COCO 평가 결과 저장 경로
    # save_hybrid=False,    # type: bool, default: False, hybrid 결과 저장 여부 (train + val 통합용)
    # save_txt=False,       # type: bool, default: False, 검출 결과를 .txt 형식으로 저장 여부
    # save_conf=False,      # type: bool, default: False, .txt 저장 시 confidence 포함 여부
    # save_crop=False,      # type: bool, default: False, 검출된 객체부분을 잘라 저장 여부
    # save_labels=False,    # type: bool, default: False, 검증 시 GT(label) 이미지 시각화 저장 여부
    # save_period=-1,       # type: int, default: -1, 몇 epoch마다 저장할지 (-1: 마지막만 저장)
    # plots=True,             # type: bool, default: True, 검증 후 confusion, matrix, PR curve 등 시각화 결과 생성
    # plots_dir='runs/val/plots',   # type: str, default: 'runs/val/plots', 시각화 결과 저장 경로
    # project='runs/val',   # type: str, default: 'runs/val', 검증 결과 저장 경로
    # name='exp',           # type: str, default: 'exp', 검증 결과 저장 폴더 이름
    # exist_ok=False,       # type: bool, default: False, 기존 폴더 덮어쓰기 허용 여부

    # 4. 시각화 옵션
    # show=False,           # type: bool, default: False, 검증 중 윈도우에 결과 출력 여부
    # show_labels=True,     # type: bool, default: True, 시각화 시 라벨 텍스트 표시 여부
    # show_conf=False,      # type: bool, default: False, 시각화 시 confidence 표시 여부
    # visualize=False,      # type: bool, default: False, feature map 시각화 여부
    # confusion_matrix=False,   # type: bool, default: False, 혼동 행렬(confusion matrix) 계산 여부 (내부적으로는 항상 계산되지만 표시 여부를 제어)

    # 5. 기타 옵션
    # verbose=False,        # type: bool, default: False, 세부 검증 로그 출력 여부
    # profile=False,        # type: bool, default: False, 각 레이어별 inference 시간 프로파일링
    # device_benchmark=False,   # type: bool, default: False, GPU 성능 벤치마크 수행 여부
    # fuse=False,           # type: bool, default: False, CONV+BN 레이어 병합 후 추론 여부
    # half=False,           # type: bool, default: False, Half precision(FP16) 추론 사용
    # dnn=False,            # type: bool, default: False, OpenCV DNN 추론 엔진 사용 여부
    # augment=False,        # type: bool, default: False, 검증 시 test-time augmentation 사용 여부
    # task='detect',        # type: str, default: 'detect', 검증 작업 유형 (detect, segment, pose, classify)
    # task_metric='mAP50-95',   # type: str, default: 'mAP50-95', 검증 시 사용될 주요 지표 (COCO 기준)
    # overlap_mask=True,    # type: bool, default: True, segmentation 시 마스크 중첩 허용 여부
    # mask_ratio=4.0,       # type: float, default: 4.0, segmentation 시 마스크 손실 비율
)
print(metrics)
