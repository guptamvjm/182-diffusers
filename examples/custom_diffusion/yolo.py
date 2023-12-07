
from ultralytics import YOLO
from PIL import Image

# returns the amount of the taller person on the left
def images_to (images, model):

    # model = YOLO('yolov8n.pt')

    results = model(images)

    # Add bounding boxes
    counter = 0
    test = []
    for result in results:

        bboxes = []
        for i in range(len(result)):
            box_obj = result[i].boxes
            xyxy = box_obj.xyxy.squeeze().cpu().detach().numpy()
            # conf = box_obj.conf
            cls = box_obj.cls
            
            if cls == 0 : #(0: humans)
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                area = (x2-x1) *(y2-y1) # Calculate area
                bboxes.append([area, (x1, y1, x2, y2)])
        
        # Compare the 2 largest bounding boxes
        comp = sorted(bboxes, key=lambda x: x[0], reverse=True)[:2]

        # Find which one is on the left
        left = 0 if (comp[0][1][2] + (comp[0][1][2] + comp[0][1][0])/2) < (comp[1][1][2] + (comp[1][1][2] + comp[1][1][0])/2) else 1

        # Check if left is taller (if y1 is greater)
        right = left^1
        
        # Increment if left is taller
        print(comp[left][1][1],  comp[right][1][1])
        test.append([comp[left][1][1],  comp[right][1][1]])
        counter += int(comp[left][1][1] < comp[right][1][1])

    return counter