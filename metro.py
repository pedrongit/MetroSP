
import time
import cv2
import numpy as np
from openvino.runtime import Core
import serial

#arduino = serial.Serial('/dev/cu.usbmodem14301', 9600) 

#arduino.write(b'1')
#PRECISAO
#precisao = "FP16"
precisao = "FP16-INT8"
#precisao = "FP32

#MODELO
#modelo = "person-detection-0200"
#modelo = "person-detection-0201"
#modelo = "person-detection-0202"
modelo = "person-detection-0203"
#modelo = "person-detection-retail-0013"


model_path = f"models/person-detection-0200/FP16-INT8/person-detection-0200.xml"           
model_weights_path = f"models/person-detection-0200/FP16-INT8/person-detection-0200.bin"

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
    # Refere-se às variáveis globais
    global points, cropping

    # Se o botão esquerdo do mouse foi clicado, registre o ponto inicial
    # e indique que estamos recortando
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        cropping = True
        print(f"Starting point registered at: {points[0]}")  # Debug line

    # Verifique se o botão esquerdo do mouse foi movido
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            points.append((x, y))
            print(f"Point registered at: {x, y}")  # Debug line

    # Verifique se o botão esquerdo do mouse foi liberado
    elif event == cv2.EVENT_LBUTTONUP:
        # Registra o ponto final
        points.append((x, y))
        cropping = False 
        print(f"Ending point registered at: {points[-1]}")  # Debug line

def draw_polygon(frame, points):
    if len(points) > 0:
        # Desenha um polígono na imagem
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

#DESENHO DA ZONA
def draw_zone(frame, pontos, boxes, opacidade=0.5):
    # Converter pontos para um array numpy
    pontos = np.array(pontos, dtype=np.int32)

    # Contar o número de centróides dentro da zona
    centroids_inside = 0
    for _, box in boxes:
        centroid_x = box[0] + box[2] // 2
        centroid_y = box[1] + box[3] // 2

        if cv2.pointPolygonTest(pontos, (centroid_x, centroid_y), False) >= 0:
            centroids_inside += 1
    cor=(0,0,255)
    # Escolher a cor com base no número de centróides dentro da zona
    if centroids_inside >= 2:
        cor = (0, 0, 255) 
        #arduino.write(b'1')
    else:
        #arduino.write(b'0')
        
        cor=(0, 255, 0)  # Vermelho se dois ou mais centróides, caso contrário verde

    espessura = 2  # Espessura da borda do polígono
    # Criar uma cópia do frame para desenhar o polígono preenchido
    filled_frame = frame.copy()
    cv2.fillPoly(filled_frame, [pontos], cor)

    # Misturar o frame original com o polígono preenchido
    cv2.addWeighted(filled_frame, opacidade, frame, 1 - opacidade, 0, frame)

    # Desenhar a borda do polígono no frame mesclado
    cv2.polylines(frame, [pontos], True, cor, espessura)

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

        # Se a tecla 'c' for pressionada, sai do modo de desenho
        elif key == ord("c"):
            break
    print (points)
    while True:
        _,frame = vs.read()
        pixel_scale = 50  # Espaçamento entre as linhas da grade em pixels
        draw_dashed_grid(frame,pixel_scale=50)
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
        
        cv2.imshow(winname="ESC pra Sair", mat=frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    vs.release()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = 0

    main(video)