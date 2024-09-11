import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from sklearn import *
import os
import time
import base64
import time 
from kafka import KafkaConsumer, KafkaProducer
import json
import threading
from align_faces import align_img
from turbojpeg import TurboJPEG
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info(f"Starting face recognition...")

# output to Kafka
KAFKA_SERVER = os.getenv('KAFKA_SERVER')
OUTPUT_DIR_PATH = os.getenv('OUTPUT_DIR_PATH')

producer = KafkaProducer(
           bootstrap_servers=[KAFKA_SERVER],
           value_serializer=lambda x: json.dumps(x).encode('utf-8')
           )

jpeg = TurboJPEG()

def boxes2xywh(box):
    x = int(min(box[0], box[2]))
    w = int(abs(box[2] - box[0]))
    y = int(min(box[1], box[3]))
    h = int(abs(box[3] - box[1]))
    return [x, y, w, h]


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)

class Arcface_trt(object):
    def __init__(self, engine_file_path, gpu_device_no, kafka_producer_topic):
        # Create a Context on this device,
        self.gpu_device_no = int(gpu_device_no)
        self.kafka_producer_topic = kafka_producer_topic
        self.cfx = cuda.Device(self.gpu_device_no).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * 1
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                logging.info(f"Binding is DOOOOOOOONE")
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                #print("NOOO")
                host_outputs.append(host_mem)
                logging.info(f"Binding is NOOOOONE")
                cuda_outputs.append(cuda_mem)
        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    #frame_path,bboxes,landmarks,sending_data
    def infer(self, image_base64, bboxes, landmarks, content_id, file_type, timestamp, cnt, count_frames, file_name):
        threading.Thread.__init__(self)
        
        embs = []
        # in_file = open(frame_path, 'rb')
        # image_raw = jpeg.decode(in_file.read())
        image_raw = self.read_base64_img(image_base64)
        # Saving frame - delete after test
        current_dir = os.path.join(OUTPUT_DIR_PATH, content_id)
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        # cv2.imwrite(os.path.join(current_dir, str(cnt)+'.jpg'), image_raw)
        
        for i in range(len(bboxes)):
            #box_xywh = boxes2xywh(bboxes[i])
            #print('land type', landmarks)
            aligned_img = align_img(image_raw, landmarks[i])
            img, img_raw = self.preprocess_image(aligned_img, i)
            emb = self.get_embedding(img)
            emb = self.post_process(emb)
            box_xywh = boxes2xywh(bboxes[i])
            # res_dict = {
            #     "x": box_xywh[0],
            #     "y": box_xywh[1],
            #     "width": box_xywh[2],
            #     "height": box_xywh[3],
            #     "vector": emb,
            # }
            res_dict = {
                "vector": emb,
                "coordinates": [box_xywh[0], 
                                box_xywh[1],
                                box_xywh[2],
                                box_xywh[3]],
                "bbox": bboxes[i]
                }
            embs.append(res_dict)
        to_kafka={'content_id': content_id, 'file_type': file_type, 'timestamp': timestamp, 'frame': image_base64, 'faces': embs, 'count_frames': count_frames, 'file_name': file_name}
        # logging.info(f"content_id': {content_id}, vector : {len(emb)}")
        # if len(bboxes)>0:
        #     #send_data=message
        #     del send_data["landmarks"]
        #     del send_data["bboxes"]
            #send_data['results'] = embs
            #sending_data['results'] = embs
            
            #print(sending_data)
        producer.send(self.kafka_producer_topic, value=to_kafka)
        logging.info(f"SENT TO KAFKA: file id: {content_id}, file type: {file_type}, timestamp: {timestamp}, faces: {len(embs)}, file_name: {file_name}")
        #print('sending res', sending_data)
        # r.rpush(settings.REDIS_OUT_TOPIC, json.dumps(send_data))
        #print('All time recognition :', time.time()-tt)

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def read_base64_img(self, base64_img):
        nparr = np.frombuffer(base64.b64decode(base64_img), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return img
    
    def preprocess_image(self, img, i):
        #img = cv2.imread(PATH + str(input_image_path)+ "_" + str(i) + ".jpg")
        resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = ((img_in.transpose((2, 0, 1)) - 127.5) * 0.0078125).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in = np.ascontiguousarray(img_in)
        return img_in, img
 
    #def get_embedding(self, context, input_image, h_input, h_output, d_input, d_output, stream, bindings):
    def get_embedding(self, input_image):
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        h_input = self.host_inputs
        d_input = self.cuda_inputs
        h_output = self.host_outputs
        d_output = self.cuda_outputs
        bindings = self.bindings
        a = time.time()
        # Allocate device memory for inputs and outputs.
        # print("input shape: ", input_image.shape)
        np.copyto(h_input[0], input_image.ravel())
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input[0], h_input[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output[0], d_output[0], stream)
        # Synchronize the stream
        stream.synchronize()
        self.cfx.pop()
        # Return the host output.
        b = time.time()-a
        #print("time to get embedding", b)
        return h_output[0]

    def post_process(self, emb):
        emb = emb.reshape(1,-1).tolist()
        emb = preprocessing.normalize(emb).flatten()
        emb = emb.tolist()
        return emb


if __name__ == "__main__":
    # Changes to receive data from Kafka instead of Redis - reading from env variables
    KAFKA_CONSUMER_TOPIC = os.getenv('KAFKA_CONSUMER_TOPIC')
    KAFKA_CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP')
    # Producer to send meta to recognition process
    KAFKA_PRODUCER_TOPIC = os.getenv('KAFKA_PRODUCER_TOPIC')

    GPU_DEVICE_NO = os.getenv('GPU_DEVICE_NO')
    ENGINE_FILE_PATH = os.getenv('ENGINE_FILE_PATH')
    arcface = Arcface_trt(ENGINE_FILE_PATH, GPU_DEVICE_NO, KAFKA_PRODUCER_TOPIC)

    while True:
        consumer = KafkaConsumer(
            KAFKA_CONSUMER_TOPIC,
            bootstrap_servers=[KAFKA_SERVER],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id=KAFKA_CONSUMER_GROUP,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')))
        cnt = 0
        for message in consumer:
            message = message.value
            image_base64 = message['image_base64']
            bboxes = message['bboxes']
            landmarks = message['landmarks']
            timestamp = message['timestamp']
            file_type = message['file_type']
            content_id = message['content_id']
            count_frames = message['count_frames']
            file_name = message['file_name']
            # Sending to thread to work in parallel
            thread = myThread(arcface.infer, [image_base64, bboxes, landmarks, content_id, file_type, timestamp, cnt, count_frames, file_name])
            thread.start()
            thread.join()
            cnt += 1

    arcface.destroy()