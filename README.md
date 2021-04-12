# [OpenVINO_Tutorial](https://github.com/DYF-AI/OpenVINO_Tutorials/)

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)

- This project mainly provides learning tutorials for beginners of OpenVINO. This project refers to the code of [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo), but we rewrite the code of [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo) to adapt to the real deployment environment

- The workflow is as follows:

  **Work flow: Pytorch-->ONNX-->OpenVINO**

- [Model visualization](https://netron.app/)


- [yolov5(v4.0)](https://github.com/ultralytics/yolov5)
  ```shell
  cd yolov5
  source script/run.sh
  pip_      # requirements
  detect    # test
  export_onnx  # onnx
