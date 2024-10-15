import numpy as np
import cv2
import face_recognition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(image_paths, labels):
    faces = []
    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        if len(face_encoding) > 0:
            faces.append(face_encoding[0])
        else:
            print(f"No face found in {image_path}")
    
    X = np.array(faces)
    y = np.array(labels)
    
    return X, y

def train_models(X_train, y_train):
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(kernel='linear', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def live_facial_recognition(models, scaler, label_names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale the face encoding
            scaled_encoding = scaler.transform([face_encoding])
            
            # Predict using each model
            predictions = {}
            for model_name, model in models.items():
                prediction = model.predict_proba(scaled_encoding)[0]
                predicted_label = label_names[prediction.argmax()]
                confidence = prediction.max()
                predictions[model_name] = (predicted_label, confidence)
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Write predictions on the frame
            y = top - 15 if top - 15 > 15 else top + 15
            for i, (model_name, (label, confidence)) in enumerate(predictions.items()):
                text = f"{model_name}: {label} ({confidence:.2f})"
                cv2.putText(frame, text, (left, y + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    # Replace these with your actual image paths and labels
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
    labels = ['person1', 'person2', ...]
    
    # Load and preprocess the data
    X, y = load_and_preprocess_data(image_paths, labels)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train models
    models = train_models(X_train_scaled, y_train)
    
    # Start live facial recognition
    live_facial_recognition(models, scaler, list(set(labels)))

if __name__ == "__main__":
    main()