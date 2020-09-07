import face_recognition
import os
from cv2 import cv2
import pickle
import time

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'

video = cv2.VideoCapture(0)

print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        encoding = pickle.load(open(f'{KNOWN_FACES_DIR}/{name}/{filename}',"rb"))

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(int(name))

if len(known_names) > 0:
    next_id = max(known_names) + 1
else:
    next_id = 0

print('Processing unknown faces...')
while True:

    # Load image
    ret,image = video.read()
    locations = face_recognition.face_locations(image, model=MODEL)

    encodings = face_recognition.face_encodings(image, locations)

    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')
        else:
            match = str(next_id)
            next_id += 1
            known_names.append(match)
            known_faces.append(face_encoding)
            os.mkdir(f'{KNOWN_FACES_DIR}/{match}')
            pickle.dump(face_encoding,open(f'{KNOWN_FACES_DIR}/{match}/{match}-{int(time.time())}.pkl',"wb"))

        # Each location contains positions in order: top, right, bottom, left
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        color = [0,100,200]

        # Paint frame
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)

        # Paint frame
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

        # Wite a name
        cv2.putText(image, str(match), (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow("", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break