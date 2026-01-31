import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("vid.mp4")
model = YOLO("yolov8n.pt")

zone = 50,100,500,700
closezone = zone[0]-100, zone[1]-100, zone[2]+100, zone[3]+100

def intersec(pers, zone):
    x1,y1,x2,y2 = pers
    zx1,zy1,zx2,zy2 = zone
    if x1 > zx2 or x2 < zx1 or y1 > zy2 or y2 < zy1:
        return False
    return True

i = 0
inside_before = False

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)

    cv2.rectangle(frame, (zone[0],zone[1]), (zone[2],zone[3]), (0,0,255), 1)

    inside_now = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            if name == "person":
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                zonepers = (x1,y1,x2,y2)

                if intersec(zonepers, zone):
                    inside_now = True
                    color = (0,0,255)
                    status = "outsider"

                    if not inside_before:
                        crop = frame[y1:y2, x1:x2]
                        cv2.imwrite("criminals/outsider_"+str(i)+".jpg", crop)
                        i += 1

                elif intersec(zonepers, closezone):
                    color = (0,255,255)
                    status = "sussy"
                else:
                    color = (0,255,0)
                    status = ""

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, status, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    inside_before = inside_now

    cv2.imshow("video", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
