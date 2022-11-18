
import time
import cv2
import numpy as np
from openvino.runtime import Core

#OpenVino
precision="FP16"
#precision="FP16-INT8"
#precision="FP32"

#PATH MODELO 0013
model_path = f"models/person-detection-retail-0013/{precision}/person-detection-retail-0013.xml"
model_weights_path = f"models/person-detection-retail-0013/{precision}/person-detection-retail-0013.bin"

ie_core = Core()
model = ie_core.read_model(model=model_path, weights=model_weights_path)
compiled_model = ie_core.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

_,_,height, width = input_layer.shape


#PROCESSAMENTO DO RESULTADO DO MODELO 
def process_boxes(frame, results, thresh=0.6):
    dim=frame.shape
    h=dim[0]
    w=dim[1]
    results = results.squeeze()
    boxes = []
    scores = []
    for image_id, label, conf,xmin, ymin, xmax, ymax in results:
        box=tuple (map(int,(xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
        boxes.append(box)

        scores.append(conf)

    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6)


    if len(indices) == 0:
        return []

    return [(scores[idx], boxes[idx]) for idx in indices.flatten()]

#DESENHO DAS CAIXAS
def draw_boxes_frame(frame, boxes):
    colors = {"red": (0, 0, 255), "green": (0, 255, 0)}
    for score, box in boxes:
        
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=colors["red"], thickness=1)
        
        
        #cv2.circle(img=frame,center=(xc,yc),radius=0,color=colors["red"],thickness=1)
        perc=score*100
        cv2.putText(
            img=frame,
            text=f"{perc:.2f}%",
            org=(box[0] + 10, box[1] + 10),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 3000,
            color=colors["green"],
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return frame

#MAIN  
def main(source):
    size = (width, height)
    vs = cv2.VideoCapture(source)
    #vw = cv2.VideoWriter('video1.avi', 
                       #  cv2.VideoWriter_fourcc(*'MJPG'),
                        # 10, size)
    
    while True:
        _,frame = vs.read()
        
        #Cria a janela
        cv2.namedWindow(winname="ESC pra Sair", flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        
        #Video acabou
        if frame is None:
            break

        #Adapta imagem pra rede     
        resized_image = cv2.resize(frame, (width, height))
        data = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    
        
        t_inicial= time.time()
        
        #
        results = compiled_model([data])[output_layer]
        boxes = process_boxes(frame=frame, results=results)
        frame = draw_boxes_frame(frame=frame, boxes=boxes)

        t_final = time.time()
        t=[]
        t.append(t_final - t_inicial)
        t_processamento = np.mean(t) * 1000
        fps = 1000 / t_processamento
        cv2.putText(
                img=frame,
                text=f"Tempo de inferencia: {t_processamento:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=frame.shape[1] / 2000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        cv2.imshow(winname="ESC pra Sair", mat=frame)
        #vw.write(frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    vs.release()
    
    cv2.destroyAllWindows()

#VIDEO OU WEBCAM
source=0

source="data/video.avi"
#source="data/video_r.avi"
#source="data/video2.avi"
main(source)