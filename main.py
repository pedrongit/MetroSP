import cv2
import numpy as np
from openvino.inference_engine import IECore

# Load the YOLOv3 OpenVINO model
model_path = "models/yolo-v3-tf/FP32/yolo-v3-tf.xml"
weights_path = "models/yolo-v3-tf/FP32/yolo-v3-tf.bin"
model_path = f"models/person-detection-retail-0013/FP16/person-detection-retail-0013.xml"
model_weights_path = f"models/person-detection-retail-0013/FP16/person-detection-retail-0013.bin"

ie = IECore()
net = ie.read_network(model=model_path, weights=weights_path)
exec_net = ie.load_network(network=net, device_name="CPU")

input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

# Get the input dimensions
_, _, height, width = net.input_info[input_blob].input_data.shape

def preprocess_image(image, input_width, input_height):
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = np.transpose(resized_image, (2, 0, 1))
    return np.expand_dims(transposed_image, 0)

def main(video_path):
    vc = cv2.VideoCapture(video_path)

    while vc.isOpened():
        ret, frame = vc.read()

        if not ret:
            break

        # Preprocess the input frame
        input_data = preprocess_image(frame, width, height)

        # Inference
        result = exec_net.infer({input_blob: input_data})
        detections = result[output_blob]

        # Loop through the detections and check if it's a person
        for detection in detections[0][0]:
            score = float(detection[2])
            class_id = int(detection[1])

            # Check if the detection is a person and the confidence is greater than 0.5
            if score > 0.5 and class_id == 1:
                x_min, y_min, x_max, y_max = (
                    int(detection[3] * frame.shape[1]),
                    int(detection[4] * frame.shape[0]),
                    int(detection[5] * frame.shape[1]),
                    int(detection[6] * frame.shape[0]),
                )

                # Draw the bounding box around the person
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Show the video
        cv2.imshow("Top Down People Detection", frame)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vc.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data/360 video, Shibuya Crossing. Tokyo, Japan. 8K video.mp4"
    main(video_path)
