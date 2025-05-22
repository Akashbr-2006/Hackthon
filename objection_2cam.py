import torch
import cv2

phone_cam_url = "http://192.168.161.164:4747/video"
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]

def detect_people_dual_camera():
    # Access the cameras
    phone_cam = cv2.VideoCapture(0)
    
    if not phone_cam.isOpened():
        print("Error: Could not access the phone camera.")
        return
    
    
    
    print("Press 'q' to exit.")
    
    while True:
        # Read frames from both cameras
        ret1, phone_frame = phone_cam.read()
        if not ret1:
            print("Failed to grab frame from phone camera.")
            break
        
       

        # Resize frames to the same dimensions for concatenation
        phone_frame_resized = cv2.resize(phone_frame, (640, 480))

        # Combine frames horizontally (side-by-side)
        combined_frame = cv2.hconcat([phone_frame_resized])

        # Perform YOLOv5 detection on the combined frame
        results = model(combined_frame, size=640)

        # Render detections on the frame
        results.render()  # Annotates the combined frame in-place

        # Ensure the annotated frame remains in BGR for OpenCV display
        annotated_frame = results.ims[0]
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Display the annotated combined frame
        cv2.imshow("Person Detection - Combined Camera Feeds", annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    phone_cam.release()
   
    cv2.destroyAllWindows()

# Run the dual-camera live feed detection
detect_people_dual_camera()
