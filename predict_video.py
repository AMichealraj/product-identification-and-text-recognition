
import cv2 as cv

import easyocr


import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Usage: python predict_video.py --input=path/to/the/video/file.mp4 --out=output.avi')
    #parser.add_argument('--input', help='Input video file.')
    #parser.add_argument('--out', help='Name of the output video file.')
    #args = parser.parse_args()

    #if not os.path.isfile(args.input):
        #print("Input video file ", args.input, " not found!")
        #sys.exit(1)

    #fourcc = cv.VideoWriter_fourcc(*"MJPG")
    #img = np.zeros((600, 600), 'uint8')
    #out = cv.VideoWriter(args.out, fourcc, 30.0, (1920,1080), True)
    #out = cv.VideoWriter(img,fourcc, 30.0, (1920, 1080), True)

    #yolo = Yolo(img_width=416, img_height=416,
                #confidence_threshold=0.85, non_max_supress_theshold=0.7,
                #classes_filename='./config/classes.names',
                #model_architecture_filename="./config/yolov3_license_plates.cfg",
                #model_weights_filename="./config/yolov3_license_plates_last.weights",
                #output_directory='./debug/',
                #output_image=False)
    
    #ocr = OCR(model_filename="./config/attention_ocr_model.pth", use_cuda=False)

    font = cv.FONT_HERSHEY_SIMPLEX
    reader = easyocr.Reader(['en'])

    cap = cv.VideoCapture(0)

    frame_count = 0
    while(cap.isOpened()):
        hasFrame, frame = cap.read()
        if hasFrame:
            frame_count += 1
            print(frame_count)
            if frame_count % 5 == 0: # process every other frame to save time
                img = frame
                result = reader.readtext(img)

                #spacer = 100

                for detection in result:
                    text = detection[1]
                    print(text)

                    speak.Speak(text)







            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            cv.imshow('frame', frame)
        else:
            break

    cv.destroyAllWindows()
    cap.release()
    #out.release()
