import cv2
from net import Net
from loader import FER2013
import torch
import numpy as np

def detect_and_draw(frame):
    ''' Processes the frame and returns it with a rectangle drawn around the detected face. '''
    haar = cv2.CascadeClassifier("assets/haarcascade_frontalface.xml")     ## use haar cascade to locate face
    faces = haar.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 10, minSize = (75, 75))
    
    ## start preprocessing for passing to neural net
    gray_frame = cv2.resize(frame, (48, 48))
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)   ## convert to gray
    dest = np.zeros((48, 48))
    gray_frame = cv2.normalize(gray_frame, dest, 1.0, 0.0, cv2.NORM_MINMAX)

    # net = Net(1, 7)
    # net.load_state_dict(torch.load('model.pt'))
    # net.eval()

    # pred = net(torch.from_numpy(gray_frame).unsqueeze(0).unsqueeze(0).float())

    ##pred = torch.argmax(pred).item()

    print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 20, 20), 3)
    return(frame)

def main():
    capture = cv2.VideoCapture(0)

    while True:
        val, frame = capture.read() ## val is true when the frame is read successfully

        if not val:
            break

        new_frame = detect_and_draw(frame) ## get the frame with rectangle drawn around the face

        cv2.imshow('Live Emotion Detector', new_frame)
        ##print(f"Emotion prediction is: {pred}")

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()