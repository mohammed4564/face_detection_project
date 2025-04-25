import cv2
import os

def generate_dataset():
    # Load Haarcascade face detector
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt_tree.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            # Optional: Draw rectangle around the face (for user feedback)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            return img[y:y + h, x:x + w]

    # Input for user ID
    try:
        user_id = int(input("Enter user ID (numeric): "))
    except ValueError:
        print("Invalid ID. Please enter a number.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    img_id = 0
    max_images = 200

    # Define the directory to save images
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("[INFO] Starting face capture. Look at the camera.")
    print("[INFO] Press 'Enter' key or wait to reach 200 images...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cropped_face = face_cropped(frame)
        if cropped_face is not None:
            img_id += 1
            face_resized = cv2.resize(cropped_face, (200, 200))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

            file_path = os.path.join(data_dir, f"user.{user_id}.{img_id}.jpg")
            cv2.imwrite(file_path, face_gray)

            cv2.putText(face_gray, f"Img {img_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Captured Face", face_gray)

            print(f"[INFO] Captured image {img_id}")

            if cv2.waitKey(1) == 13 or img_id >= max_images:  # Enter key
                break
        else:
            print("[INFO] No face detected. Try again...")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Dataset collection completed.")

if __name__ == "__main__":
    generate_dataset()
