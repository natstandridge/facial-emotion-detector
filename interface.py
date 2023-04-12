import cv2

def detect_and_draw(frame):
    ''' Processes the frame and returns it with a rectangle drawn around the detected face. '''
    haar = cv2.CascadeClassifier("assets/haarcascade_frontalface.xml")     ## use haar cascade to locate face
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                   ## convert to gray

    frame = cv2.resize(frame, (48, 48))

    faces = haar.detectMultiScale(gray_frame, scaleFactor = 1.1, minNeighbors = 10, minSize = (75, 75))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 20, 20), 3)
    return(frame)

def main():
    capture = cv2.VideoCapture(0)

    while True:
        val, frame = capture.read() ## val is true when the frame is read successfully

        if not val:
            break

        frame_with_box = detect_and_draw(frame) ## get the frame with rectangle drawn around the face

        cv2.imshow('Facial Emotion Detector', frame_with_box)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    capture.release()
    cv2.destroyAllWindows()

main()