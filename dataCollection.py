import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # kameraAyarı
# kernel = np.ones((12, 12), np.uint8)
detector = HandDetector(maxHands=1)  # algılananElSayısı

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) # img'de gordugu eli tanıyıp eklemler arası dogru olusturur. Sag-Sol el farkındalıgı vardır.
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]  # uzunlukTanımlamaları

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # beyazPencereÖlcegi

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # elLokasyonu

        # imgCropGrey = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
        # imgCropHSV = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
        #
        # lowerValue = np.array([0, 48, 80])
        # highValue = np.array([20, 255, 255])
        #
        # colorFilterResult = cv2.inRange(imgCropHSV, lowerValue, highValue)
        # colorFilterResult = cv2.morphologyEx(colorFilterResult, cv2.MORPH_CLOSE, kernel)

        imgCropShape = imgCrop.shape

        aspectRatio = h / w  # yükseklik/Genislik

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)  # anlikWhesaplar
            imgReSize = cv2.resize(imgCrop, (wCal, imgSize))  # imgYeniBoyut
            imgReSizeShape = imgReSize.shape
            wGap = math.ceil((imgSize - wCal) / 2)  # KoselereOlanWboslugu
            imgWhite[:, wGap:wCal + wGap] = imgReSize  # beyazPencereİcineEliKırpıpAtma.#elLokasyonu

            # beyazPenceredeGenislikOturuTasmaDurumundaAppDurmaz.

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)  # anlikWhesaplar
            imgReSize = cv2.resize(imgCrop, (imgSize, hCal))  # imgYeniBoyut
            imgReSizeShape = imgReSize.shape
            hGap = math.ceil((imgSize - hCal) / 2)  # koselereOlanWboslugu
            imgWhite[hGap:hCal + hGap, :] = imgReSize  # beyazPencereİcineEliKırpıpAtma.#elLokasyonu


            # beyazPenceredeYukseklikOturuTasmaDurumundaAppDurmaz.


        cv2.imshow("ImageCrop", imgCrop)  # kucukPencereElBoyutunaGore
        cv2.imshow("ImageWhite", imgWhite)  # beyazPencere
        # cv2.imshow("colorFilterResult", imgCropGrey)

    cv2.imshow("Camera", img)  # pencereAdıveİçerigi
    key = cv2.waitKey(1)  # videolaştırma
    if key == ord("s"): # s tusuna basılırsa
        counter += 1  # sayacı(counter) 1 arttır
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)  # imgWhite penceresi jpg türünde ilgili path'e atılır.
        print(counter)
