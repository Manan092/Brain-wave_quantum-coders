
import cv2
from ultralytics import YOLO
import time


model = YOLO("yolov8x.pt")


genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)


mean_value = [78.4263377603, 87.7689143744, 114.895847746]
gender_list = ['Female', 'Male']


source = "vid10.mp4"  


vid_capture = cv2.VideoCapture(source)


fps = vid_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(vid_capture.get(3))
frame_height = int(vid_capture.get(4))


out = cv2.VideoWriter('output17.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


class_names = model.names


person_class_index = None
for class_id, class_name in class_names.items():
    if class_name == 'person':
        person_class_index = class_id
        break


male_count = 0
female_count = 0
gender_counts = {'Male': 0, 'Female': 0}

frame_count=0
last_displayed_time = -1

while True:
    ret, frame = vid_capture.read()
    if not ret:
        break
    
    start_time = time.time()

    
    results = model(frame)
    boxes = results[0].boxes  # Get the bounding boxes

   
    male_count = 0
    female_count = 0

   
    for box in boxes:
        cls = int(box.cls)  # Class ID
        if cls == person_class_index:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
            confidence = float(box.conf)  # Convert confidence score to float

            
            face = frame[y1:y2, x1:x2]

            
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean_value, swapRB=False)
            genderNet.setInput(face_blob)
            gender_preds = genderNet.forward()
            gender = gender_list[gender_preds[0].argmax()]

            
            if gender == 'Male':
                male_count += 1
            else:
                female_count += 1

            
            label = f"{gender}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 129), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    
    gender_counts['Male'] = male_count
    gender_counts['Female'] = female_count


    text_time=int(frame_count/fps)
    if text_time%5==0 and text_time!=last_displayed_time:
        last_displayed_time=text_time
        cv2.putText(frame, f"Males: {male_count}", (10, frame_height - 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, f"Females: {female_count}", (frame_width - 150, frame_height - 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 0), 2)


   
    out.write(frame)

    
    cv2.imshow("Person and Gender Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    elapsed_time = time.time() - start_time
    time.sleep(max(1./fps - elapsed_time, 0))
    frame_count+=1


vid_capture.release()
out.release()
cv2.destroyAllWindows()
