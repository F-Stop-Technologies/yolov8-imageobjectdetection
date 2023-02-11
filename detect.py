import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os

def browseFiles():
    file_path = filedialog.askopenfilename()
    detectObjects(file_path)

def detectObjects(file_path):
    # Load the YOLOv8 model
    net = cv2.dnn.readNet("yolov8n.pt", "ultralytics/yolo/cfg/default.yaml")

    # Load the input image
    image = cv2.imread(file_path)

    # Get the height and width of the image
    (H, W) = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Perform a forward pass through the network
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    # Initialize lists to store the bounding boxes and associated confidences scores
    boxes = []
    confidences = []

    # Loop over the layer outputs
    for output in layer_outputs:
        # Loop over each detection
        for detection in output:
            # Extract the class ID and confidence (probability) of the prediction
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections by ignoring detections with a confidence less than a minimum threshold
            if confidence > 0.5:
                # Scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates and associated confidences scores
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[class_id]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_id], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Display the output image
            cv2.imshow("Image", image)
            cv2.waitKey(0)
#Initialize a list of colors to represent each possible class label
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))
#Create the GUI window
root = tk.Tk()
root.title("Object Detection")
#Create a button to browse for files
browse_button = tk.Button(root, text="Browse", command=browseFiles)
browse_button.pack(pady=10)
#Run the gui
root.mainloop()
