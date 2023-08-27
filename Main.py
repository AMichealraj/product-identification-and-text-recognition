
import numpy as np

from tkinter import *
import os
from tkinter import filedialog
import cv2
import time





def endprogram():
	print ("\nProgram terminated!")
	sys.exit()









def Object():
    import numpy as np
    import argparse
    import cv2 as cv
    import subprocess
    import time
    import os
    from yolo_utils import infer_image, show_image

    FLAGS = [0]

    labels = open('yolov3-coco/coco-labels').read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(os.path.abspath("yolov3-coco/yolov3.cfg"),
                                    os.path.abspath("yolov3-coco/yolov3.weights"))

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    count = 0

    vid = cv.VideoCapture(0)
    while True:
        _, frame = vid.read()
        height, width = frame.shape[:2]

        if count == 0:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors,
                                                                    labels, FLAGS=False)
            count += 1
        else:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors,
                                                                    labels, FLAGS, boxes,
                                                                    confidences, classids, idxs, infer=False)
            count = (count + 1) % 6

        # text = "{}: {:4f}".format(labels[classids[i]], confidences[i])

        cv.imshow('webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()










def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.title("Blind People Assistance")

    Label(text="Blind People Assistance", width="300", height="5", font=("Calibri", 16)).pack()


    Button(text="Object Detection", font=(
        'Verdana', 15), height="2", width="30", command=Object, highlightcolor="black").pack(side=TOP)

    Label(text="").pack()


    Label(text="").pack()

    main_screen.mainloop()


main_account_screen()

