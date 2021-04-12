# export scirpt/run.sh
pip_()
{
    pip install -r ../requirements.txt
}

detect()
{
    # python3 detect.py --source data/images --weights weights/yolov5s.pt --conf 0.25
    # docker 中调用摄像头
    python3 detect.py --source 0 --weights weights/yolov5s.pt
    # 视频文件，并保存
    # python3 detect.py --source inference/*.mp4 --weights weights/yolov5s.pt --output inference/output/output.mp4
}

export_onnx()
{
    python3 models/export.py --weights weights/yolov5s.pt
}