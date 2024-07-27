import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from sklearn import *
import os
import time
import csv
import time 
from kafka import KafkaConsumer, KafkaProducer
import json
import threading
from align_faces import align_img
from turbojpeg import TurboJPEG
from redis import Redis
import settings

# output Redis
r = Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0) 

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
    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(1).make_context()
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
                print("YEEES")
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                #print("NOOO")
                host_outputs.append(host_mem)
                print("NOOO")
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
    def infer(self, frame_path, bboxes, landmarks,message):
        threading.Thread.__init__(self)
        
        embs = []
        in_file = open(frame_path, 'rb')
        image_raw = jpeg.decode(in_file.read())
        
        for i in range(len(bboxes)):
            #box_xywh = boxes2xywh(bboxes[i])
            #print('land type', landmarks)
            aligned_img = align_img(image_raw, landmarks[i])
            img, img_raw = self.preprocess_image(aligned_img, i)
            emb = self.get_embedding(img)
            emb = self.post_process(emb)
            box_xywh = boxes2xywh(bboxes[i])
            res_dict = {
                "x": box_xywh[0],
                "y": box_xywh[1],
                "width": box_xywh[2],
                "height": box_xywh[3],
                "vector": emb,
            }
            embs.append(res_dict)
        send_data=message
        #del send_data["landmarks"]
        #del send_data["bboxes"]
        send_data['results'] = embs
        if len(bboxes)>0:
            #send_data=message
            del send_data["landmarks"]
            del send_data["bboxes"]
            #send_data['results'] = embs
            #sending_data['results'] = embs
            
            #print(sending_data)
        #print('sending res', sending_data)
        r.rpush(settings.REDIS_OUT_TOPIC, json.dumps(send_data))
        #print('All time recognition :', time.time()-tt)

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

   
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
        print("input shape: ", input_image.shape)
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

arcface = Arcface_trt("build/arcface-r100.engine")

KAFKA_SERVER = os.getenv('KAFKA_SERVER')

while True:
    consumer = KafkaConsumer(
        settings.KAFKA_CONSUMER_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='my-group', # Change group name
        value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    
    print(settings.KAFKA_CONSUMER_TOPIC)
    
    for message in consumer:
        #print(message)
        message = message.value
        print(message)
        frame_path = message['frame_path']
        bboxes = message['bboxes']
        landmarks = message['landmarks']
        #sending_data = {
        #"frame_path": frame_path,                    
        #"camera_id":message['camera_id'],
        #"timestamp":message['timestamp'],                    
        #"face":message['face'],                    
        #"license_plate":message['license_plate'],                    
        #"person_count":message['person_count'],                    
        #"vehicle_count":message['vehicle_count'],
        #}
        #print(message)
        thread = myThread(arcface.infer, [frame_path,bboxes,landmarks,message])
        thread.start()
        #print(res_kafka)
        thread.join()


arcface.destroy()

#img2 = cv2.imread("./align_3.jpg")
#emb2 = get_embedding(img2)

#print(compute_sim(emb1, emb2))
#print("sim: ", similarity(emb1,emb2))
