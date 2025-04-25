import cv2
import os

def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        # Only return the first detected face (if you expect multiple faces, adjust accordingly)
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    id = 1
    img_id = 0

    # Define the directory to save images using a raw string to handle backslashes
    data_dir = r"C:\Users\lenovo\OneDrive - DATANOVELTECH PRIVATE LIMITED\Desktop\Project folders\face_recognization_project\data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)  # Create the directory if it doesn't exist

    print("Starting the capture. Press 'Enter' to stop or capture 200 images.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cropped_face = face_cropped(frame)
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Define the file path
            file_name_path = os.path.join(data_dir, f"user.{id}.{img_id}.jpg")
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Show the cropped face in a window
            cv2.imshow("Cropped Face", face)
            
            # Wait for a key press, and close the window if Enter is pressed or capture 200 images
            key = cv2.waitKey(1)
            if key == 13 or img_id >= 200:  # 13 is the ASCII code for Enter
                break
        else:
            print("No face detected.")

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting sample images is completed")

generate_dataset()
