
import time
import cv2
import numpy as np
from openvino.runtime import Core
import serial

arduino = serial.Serial('COM4', 9600) 

#arduino.write(b'1')
#PRECISAO
#precisao = "FP16"
precisao = "FP16-INT8"
#precisao = "FP32

#MODELO
#modelo = "person-detection-0200"
#modelo = "person-detection-0201"
#modelo = "person-detection-0202"
#modelo = "person-detection-0203"
#modelo = "person-detection-retail-0013"


model_path = f"models/person-detection-retail-0013/FP16-INT8/person-detection-retail-0013.xml"           
model_weights_path = f"models/person-detection-retail-0013/FP16-INT8/person-detection-retail-0013.bin"

print("Model XML path:", model_path)
print("Model BIN path:", model_weights_path)


ie_core = Core()
model = ie_core.read_model(model=model_path, weights=model_weights_path)
compiled_model = ie_core.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

_,_,height, width = input_layer.shape

points = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global points, cropping
    #Se LeftButton for apertado salva o ponto na variavel global
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:  
            points.append([x, y])
            cropping = True
            print(f"Point {len(points)}: {points[-1]}")  # Debug line
        else:
            print("Four points already selected. Press 'r' to reset points if needed.")


def draw_polygon(frame, points):
    if len(points) == 4:  # Only draw the polygon if we have exactly four points
        # Draw a polygon on the image
        cv2.polylines(frame, [np.array(points)], True, (0, 255, 0), thickness=2)
    return frame


def draw_dashed_line(frame, pt1, pt2, color, thickness, dash_length):
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    num_dashes = int(dist / (2 * dash_length))
    pts = np.linspace(pt1, pt2, num_dashes * 2, dtype=int)
    for i in range(0, len(pts) - 1, 2):
        cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)


def draw_dashed_grid(frame, pixel_scale, dash_length=5):
    height, width, _ = frame.shape
    color = (255, 255, 255)  # Cor das linhas da grade (branco)
    thickness = 1  # Espessura das linhas da grade

    # Desenhar linhas verticais pontilhadas
    for x in range(0, width, pixel_scale):
        draw_dashed_line(frame, (x, 0), (x, height), color, thickness, dash_length)

    # Desenhar linhas horizontais pontilhadas
    for y in range(0, height, pixel_scale):
        draw_dashed_line(frame, (0, y), (width, y), color, thickness, dash_length)


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
    colors = {"red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    for score, box in boxes:
        
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=colors["red"], thickness=1)
        
        centroid_x = box[0] + box[2] // 2
        centroid_y = box[1] + box[3] // 2
        
        # Draw the centroid
        cv2.circle(img=frame, center=(centroid_x, centroid_y), radius=2, color=colors["white"], thickness=-1)

        

        # perc=score*100
        # cv2.putText(
        #     img=frame,
        #     text=f"{perc:.2f}%",
        #     org=(box[0] + 10, box[1] + 10),
        #     fontFace=cv2.FONT_HERSHEY_COMPLEX,
        #     fontScale=frame.shape[1] / 3000,
        #     color=colors["green"],
        #     thickness=1,
        #     lineType=cv2.LINE_AA,
        # )
    return frame

#DESENHO DA ZONA
def draw_zone(frame, points, boxes, opacidade=0.5):
    # Convert points to a numpy array
    points = np.array(points, dtype=np.int32)

    # Count the number of centroids inside the zone
    centroids_inside = 0
    for _, box in boxes:
        centroid_x = box[0] + box[2] // 2
        centroid_y = box[1] + box[3] // 2

        if cv2.pointPolygonTest(points, (centroid_x, centroid_y), False) >= 0:
            centroids_inside += 1

    color = (0, 255, 0)  # Green if two or more centroids, else red

    # Choose the color based on the number of centroids inside the zone
    if centroids_inside >= 2:
        color = (0, 0, 255)
        arduino.write(b'1')
        print('Status: porta TRANCADA')
    else:
        arduino.write(b'0')
        print('Status: porta ABERTA') 

    thickness = 2  # Thickness of the polygon border
    # Create a copy of the frame to draw the filled polygon
    filled_frame = frame.copy()
    cv2.fillPoly(filled_frame, [points], color)

    # Blend the original frame with the filled polygon
    cv2.addWeighted(filled_frame, opacidade, frame, 1 - opacidade, 0, frame)

    # Draw the border of the polygon on the merged frame
    cv2.polylines(frame, [points], True, color, thickness)

    # Show zone information
    cv2.putText(frame, f"Pessoas na zona: {centroids_inside}", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#MAIN  
def main(source):
    global points
    size = (width, height)
    
    vs = cv2.VideoCapture(source)

    cv2.namedWindow("ESC pra Sair")
    cv2.setMouseCallback("ESC pra Sair", click_and_crop)
    
    _, frame = vs.read()

    while True:
        temp = frame.copy()
        draw_dashed_grid(frame,pixel_scale=50)
        draw_polygon(temp, points)
        cv2.imshow("ESC pra Sair", temp)
        key = cv2.waitKey(1) & 0xFF

        # Se a tecla 'r' for pressionada, reinicia a zona de desenho
        if key == ord("r"):
            points = []
            print(points)
        # Se a tecla 'c' for pressionada, sai do modo de desenho
        elif key == ord("c"):
            break
    print (points)
    while True:
        _,frame = vs.read()
        pixel_scale = 50  # Espaçamento entre as linhas da grade em pixels
        draw_dashed_grid(frame,pixel_scale=pixel_scale)
        frame = draw_polygon(frame, points)

        if frame is None:
            break

        #Adapta imagem pra rede     
        resized_image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        #print(f"Resized shape:{resized_image.shape}")
        data = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        #print(f"data shape:{data.shape}")


        
    
        t_inicial= time.time()
        
        #
        results = compiled_model([data])[output_layer]
        boxes = process_boxes(frame=frame, results=results)
        frame = draw_boxes_frame(frame=frame, boxes=boxes)

        #zone_points = [(0, 0), (1000, 0), (1000, 1000), (0,1000 )]  # Exemplo de coordenadas da zona (um quadrilátero) a->b->c->d
        draw_zone(frame, points, boxes, opacidade=0.5)  # Defina a opacidade desejada (0.5 neste exemplo)

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
        # Get the screen resolution
        screen_res = (1440, 900)  # Replace with your screen resolution

        # Calculate the scaling factor
        scale_factor = min(screen_res[0] / frame.shape[1], screen_res[1] / frame.shape[0])

        # Calculate the new size for the frame
        new_size = (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor))

        # Create a blank black frame of the new size
        blank_frame = np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8)

        # Resize the frame to the new size
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        # Calculate the top-left corner coordinates for placing the frame on the blank frame
        top = int((screen_res[1] - new_size[1]) / 2)
        left = int((screen_res[0] - new_size[0]) / 2)

        # Place the frame on the blank frame
        blank_frame[top:top + new_size[1], left:left + new_size[0]] = frame

        cv2.imshow(winname="ESC pra Sair", mat=blank_frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    vs.release()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = "data/video.avi"
    main(0)