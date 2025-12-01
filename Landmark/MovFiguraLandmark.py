import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(0)

fig_x, fig_y = 300, 300

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar manos
    results = hands.process(frame_rgb)

    # Dibujar los puntos clave y conexiones
    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

            x_tip, y_tip = int(index_tip.x * w), int(index_tip.y * h)
            x_mcp, y_mcp = int(index_mcp.x * w), int(index_mcp.y * h)

            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x_thumb_mcp, y_thumb_mcp = int(thumb_mcp.x * w), int(thumb_mcp.y * h)

            finger_up = y_tip < y_mcp
            thumb_up_iz = x_thumb < x_thumb_mcp

            if finger_up:
                cv2.circle(frame, (x_tip, y_tip), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "Dedo indice levantado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if thumb_up:
                    cv2.circle(frame, (x_thumb, y_thumb), 15, (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, "Dedo pulgar levantado", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "Dedo indice bajado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar la imagen
    cv2.imshow("Salida", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()