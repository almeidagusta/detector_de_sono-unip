#Código desenvolvido por: Gustavo Almeida, Júlia Vitória, Jackson Souza e Thierry Lacerda.
#Detector de Sono - Projeto Final - UNIP

# importe as bibliotecas necessárias
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import dlib
import cv2
import time

ALARM = "sonolencia_alarm.wav" #caminho para o arquivo de som
WEBCAM = 0 #índice da câmera

#função de play do alarme
def sound_alarm(path=ALARM):
	# toque o alarme
	playsound.playsound(ALARM)

#função para cálculo do EAR (abertura dos olhos)
def eye_aspect_ratio(eye):
	# computa a distância dos pontos verticais
	# coordenadas dos pontos da região dos olhos (x, y)
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# computa a distância dos pontos horizontais
	# coordenadas dos pontos da região dos olhos (x, y)
	C = dist.euclidean(eye[0], eye[3])
	# computa o EAR 
	ear = (A + B) / (2.0 * C)
	# retorna o EAR
	return ear


# definir duas constantes, uma para a proporção do olho para indicar o
# piscar dos olhos e, em seguida, uma segunda constante para o número de consecutivos
# de frames o olho deve estar abaixo do threshold para disparar o alarme
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 10
#inicialize o contador de frames e uma contante booleana para indicar o estado do alarme
COUNTER = 0
ALARM_ON = False

# initiliazaação do preditor de faces da dlib (baseado no HOG)
# preditor de pontos faciais
print("[INFO] Carregando preditor de faces...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("eye_predictor.dat") #preditor de pontos da região dos olhos (12 pontos)

# inicializa o vídeo
print("[INFO] Inicializando câmera...")
vs = VideoStream(src=WEBCAM).start()
#"aquecendo câmera"
time.sleep(2.0)

# loop sobre os frames de vídeo
while True:
    # pega o frame do vídeo, redimensiona e converte o vídeo para grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detecta faces no vídeo
	rects = detector(gray, 0)
	# loop sobre as faces  detectadas
	for rect in rects:
		# determina os pontos da região dos olhos e converte as coordenadas para um array numpy
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
        
		# extraí as coordenadas dos olhos e computa o EAR
		leftEye = shape[6:12]
		rightEye = shape[0:6]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# média do EAR dos dois olhos
		ear = (leftEAR + rightEAR) / 2.0

		# computa o fecho convexo para os pares e visualiza cada um dos olhos
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# verifica se o EAR está abaixo do EYE_AR_THRESH
		# threshold, se sim, incrementa o contador de frames
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			# se os olhos estiverem fechados por mais frames que o aceito, soe o alarme
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# se o alarme está desligado, ligar o alarme
				if not ALARM_ON:
					ALARM_ON = True
                    #inicialize uma nova thread e toque o alarme no background
					t =Thread(target=sound_alarm)
					t.deamon = True
					t.start()
					# desenha um alerta no frame
				cv2.putText(frame, "ALERTA DE SONO!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# senão, o contador é zerado e o alarme resetado
		else:
			COUNTER = 0
			ALARM_ON = False

		# desenha no frame o contador do EAR, para ajudar na depuração.		
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
 
	# apresenta o frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# a tecla "q" desliga o dispositivo
	if key == ord("q"):
		break
# fecha todas as janelas
cv2.destroyAllWindows()
vs.stop()

