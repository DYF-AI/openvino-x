import cv2
import torch
import time
import numpy as np
import torchvision
import heapq  
import onnxruntime as ort
from openvino.inference_engine import IECore

def get_label():
    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def get_model(model="mobilenet_v2"):
    if model=="mobilenet_v2":
        model = torchvision.models.mobilenet_v2(pretrained=True).eval()
    return model

def preprocess(image_path:str, resize_hw=(224, 224), return_tensor=False, device='cpu'):
    image_src = cv2.imread(image_path)
    image = cv2.resize(image_src, (resize_hw[0], resize_hw[1]))
    image = np.float32(image) / 255.0
    image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    image = image.transpose((2, 0, 1))
    if return_tensor:
        image = torch.from_numpy(image).cpu()
        image = image.unsqueeze_(0)
    return image, image_src

def post_process(res):  # res是二维数组
    label_index = np.argmax(res, 1)
    topk=[]
    for i in range(len(res)):
        b=heapq.nlargest(3, range(len(res[i])), res[i].take)
        topk.append(b)
    print("label_index:", label_index)
    print("top k:", topk[0])
    return topk

class MobileNet:
    def __init__(self, mobilenet_model="mobilenet_v2"):
        self.model = get_model(mobilenet_model)
        self.label = get_label()
       
    def predict(self, image_path, resize_hw=(224, 224)):
        image, image_src = preprocess(image_path, return_tensor=True)
        start = time.time()
        with torch.no_grad():
            output = self.model(image)
        inf_end1 = time.time() - start
        print("infer Pytorch time(ms) : %.3f" % (inf_end1 * 1000))
        res = output.detach().numpy()
        process_start = time.time()
        topk = post_process(res)
        print("Pytorch process time(ms) : %.3f" % ((time.time()-process_start)*1000))
        end = time.time()

        total_time = end - start
        inf_time_message = "Inference time: {:.3f} ms, FPS:{:.3f}".format(total_time * 1000, 1000 / (total_time * 1000 + 1))
        print(inf_time_message)
        for i in range(len(topk[0])):
            label_txt = self.label[topk[0][i]]
            cv2.putText(image_src, label_txt, (10, 100+30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
        cv2.putText(image_src, inf_time_message, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, 8)
        cv2.imwrite("pytorch.jpg", image_src)

class MobileNet_export:
    def __init__(self, mobilenet_model="mobilenet_v2", onnx_save_path="mobilenet_v2.onnx"):
        self.model = get_model(mobilenet_model)
        self.onnx_save_path = onnx_save_path
    
    def export_onnx(self):
        input_names = ['input']
        output_names = ['output']
        dummy_input = torch.randn((1, 3, 224, 224))
        torch.onnx.export(self.model, dummy_input, self.onnx_save_path, verbose=True, input_names=input_names, output_names=output_names)

class MobileNet_OnnxRT:
    def __init__(self, mobilenet_model="mobilenet_v2", onnx_path="mobilenet_v2.onnx"):
        self.onnx_path = onnx_path
        self.onnx_session()
        self.label = get_label()
    
    def onnx_session(self):
        self.ort_session = ort.InferenceSession(self.onnx_path)

    def predict_onnx(self, image_path, resize_hw=[224, 224]):
        image, image_src = preprocess(image_path, return_tensor=False)
        start = time.time()
        output = self.ort_session.run(None, {'input': [image]})
        output = output[0]
        #print("output", output)
        inf_end1 = time.time() - start
        print("infer OnnxRuntime time(ms) : %.3f" % (inf_end1 * 1000))

        process_start = time.time()
        topk = post_process(output)
        print("OnnxRuntime process time(ms) : %.3f" % ((time.time()-process_start)*1000))
        end = time.time()

        total_time = end - start
        inf_time_message = "Inference time: {:.3f} ms, FPS:{:.3f}".format(total_time * 1000, 1000 / (total_time * 1000 + 1))
        print(inf_time_message)
        for i in range(len(topk[0])):
            label_txt = self.label[topk[0][i]]
            cv2.putText(image_src, label_txt, (10, 100+30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
        cv2.putText(image_src, inf_time_message, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, 8)
        cv2.imwrite("onnx.jpg", image_src)
 
class transform:
    def __init__(self, onnx_path, ir_save_path):
        self.onnx_path = onnx_path
        self.ir_save_path = ir_save_path
        self.dynamic_input = dynamic_input
        
    def onnx_to_ir(self):
        os.system('python3.6 /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py \
                    -m {} \
                    -o {} \
                    --data_type FP32 \
                    - -input_shape[1, 3, 224, 224] '.format(self.onnx_path, self.ir_save_path))
                    
class Mobile_OpenVINO:
    def __init__(self, model_path, use_onnx=True):
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.label = get_label()
        self.init_vino()

    def init_vino(self):
        print("Creating Inference Engine!")
        self.ie = IECore()
        load_time_start = time.time() 
        self.net = self.ie.read_network(model=self.model_path)
        load_time_end = time.time()
        load_time = load_time_end - load_time_start
        print("read network time(ms) : %.3f" % (load_time * 1000))
        
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))

        start_load = time.time()
        self.exec_net = self.ie.load_network(network=self.net, device_name="CPU")
        end_load = time.time() - start_load
        print("load  time(ms) : %.3f" % (end_load * 1000))

        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape

    def predict_vino(self, image_path):
        image, image_src = preprocess(image_path, return_tensor=False)
        inf_start1 = time.time()
        res = self.exec_net.infer(inputs={self.input_blob: [image]})
        inf_end1 = time.time() - inf_start1
        res = res[self.out_blob]
        print("infer vino time(ms) : %.3f" % (inf_end1 * 1000))
        process_start = time.time()
        topk = post_process(res)
        print("vino time(ms) : %.3f" % ((time.time()-process_start)*1000))
        end = time.time()

        total_time = end - inf_start1
        inf_time_message = "Inference time: {:.3f} ms, FPS:{:.3f}".format(total_time * 1000, 1000 / (total_time * 1000 + 1))
        print(inf_time_message)
        for i in range(len(topk[0])):
            label_txt = self.label[topk[0][i]]
            cv2.putText(image_src, label_txt, (10, 100+30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
        cv2.putText(image_src, inf_time_message, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, 8)
        cv2.imwrite("vino.jpg", image_src)
 
def MobileNet_demo():
    image_path = "car.jpg"
    net = "mobilenet_v2"
    mobilenet = MobileNet(mobilenet_model=net)   
    output = mobilenet.predict(image_path)

def MobileNet_export_demo():
    mobilenet_model = "mobilenet_v2"
    onnx_save_path = "/mnt/g/models/others/mobilenet/onnx_model/mobilenet_v2.onnx"
    
    export = MobileNet_export(mobilenet_model, onnx_save_path)
    export.export_onnx()

def MobileNet_OnnxRT_demo():
    image_path = "car.jpg"
    onnx_path = "/mnt/g/models/others/mobilenet/onnx_model/mobilenet_v2.onnx"
    mobilenet = "mobilenet_v2"
    mobilenet_onnxrt = MobileNet_OnnxRT(mobilenet_model=mobilenet, onnx_path=onnx_path)   
    output = mobilenet_onnxrt.predict_onnx(image_path)

def Mobile_OpenVINO_demo(model="vino"):
    image_path = "car.jpg"
    onnx_path = "/mnt/g/models/others/mobilenet/onnx_model/mobilenet_v2.onnx"
    model_xml = "/mnt/g/models/others/mobilenet/ir_model/model_dynamic_False.xml"
    mobilenet = "mobilenet_v2"
    if model == "vino":
        print("model_type: ir")
        mobilenet_vino = Mobile_OpenVINO(model_path=model_xml, use_onnx=False)
    else:
        print("model_type: onnx")
        mobilenet_vino = Mobile_OpenVINO(model_path=onnx_path, use_onnx=True)
    output = mobilenet_vino.predict_vino(image_path)

def run_all():
    MobileNet_demo()
    print("------------------------")
    #MobileNet_export_demo()
    MobileNet_OnnxRT_demo()
    print("------------------------")
    Mobile_OpenVINO_demo(model="vino")

if __name__ == "__main__":
    import fire
    fire.Fire()
