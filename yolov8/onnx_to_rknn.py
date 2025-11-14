from rknn.api import RKNN

ONNX_PATH = '/home/linux/yolov8_ws/best.onnx'
RKNN_OUT  = '/home/linux/yolov8_ws/open_dataset_test2.rknn'

r = RKNN(verbose=True)

# 1) config: reorder_channel 대신 quant_img_RGB2BGR 사용
r.config(
    target_platform='rk3588',
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],    # [0-255] 스케일로 맞출 때 흔히 씀
    quant_img_RGB2BGR=True,          # 내부에서 RGB↔BGR 스왑
    optimization_level=3,
    float_dtype='float16'            # 선택: 속도/메모리 절충(f32 쓰려면 이 줄 지워도 됨)
)

# 2) ONNX 로드
assert r.load_onnx(model=ONNX_PATH) == 0

# 3) 빌드 (양자화 안 함: float)
assert r.build(do_quantization=False) == 0

# 4) rknn 내보내기
assert r.export_rknn(RKNN_OUT) == 0
r.release()
print('EXPORTED:', RKNN_OUT)




