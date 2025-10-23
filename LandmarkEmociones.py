import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(234, 255, 233))

cap = cv2.VideoCapture(0)

LIP_TOP = 13
LIP_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

LEFT_BROW_INNER = 70
RIGHT_BROW_INNER = 300

LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362


def get_pixel_dist(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )


            lm = face_landmarks.landmark 
            
            # Usamos la distancia entre los ojos como "unidad" base, por si afecta el zoom
            inter_eye_dist = get_pixel_dist(lm[LEFT_EYE_INNER], lm[RIGHT_EYE_INNER])
            if inter_eye_dist == 0: continue

            # --- Métricas ---

            #Enojo: Cejas bajas y juntas
            # Distancia entre cejas 
            inter_brow_ratio = get_pixel_dist(lm[LEFT_BROW_INNER], lm[RIGHT_BROW_INNER]) / inter_eye_dist
            # Distancia ceja-ojo 
            brow_eye_ratio = (get_pixel_dist(lm[LEFT_BROW_INNER], lm[LEFT_EYE_INNER]) + 
                              get_pixel_dist(lm[RIGHT_BROW_INNER], lm[RIGHT_EYE_INNER])) / 2 / inter_eye_dist

            # Felicidad: Comisuras anchas
            horizontal_mouth_ratio = get_pixel_dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT]) / inter_eye_dist

            # Tristeza/Felicidad: Comisuras de la boca. Y va de 0 [arriba] a 1 [abajo]
            corner_y = (lm[MOUTH_LEFT].y + lm[MOUTH_RIGHT].y) / 2
            top_lip_y = lm[LIP_TOP].y
            bottom_lip_y = lm[LIP_BOTTOM].y

            #print(f"Ratio Cejas Juntas: {inter_brow_ratio:.2f} | Ratio Cejas Bajas: {brow_eye_ratio:.2f} | Ratio Boca Ancha: {horizontal_mouth_ratio:.2f}")
            
            emocion = None
            color = (255, 255, 255)

            # Enojo (cejas juntas y bajas)
            if inter_brow_ratio < 3.3 and brow_eye_ratio < 1.20:
                emocion = "Enojo"
                color = (0, 0, 255)
            
            # Tristeza (labio abajo)
            elif corner_y > bottom_lip_y:
                emocion = "Tristeza"
                color = (255, 100, 0)
            
            # Felicidad (boca ancha o labio arriba)
            elif horizontal_mouth_ratio > 1.7 or corner_y < top_lip_y:
                emocion = "Felicidad"
                color = (0, 255, 0)


            # Resultado
            x_min = int(lm[MOUTH_LEFT].x * width) - 30
            y_min = int(lm[LEFT_BROW_INNER].y * height) - 40 # Texto
            
            cv2.putText(frame, emocion, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('Reconocimiento por Puntos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()