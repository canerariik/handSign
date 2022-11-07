import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)  # kameraAyarı
# kernel = np.ones((12, 12), np.uint8)
detector = HandDetector(maxHands=1)  # algılananElSayısı
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")  # teaclableMachine ile edilen data format dosya uzantısı

offset = 20  # kenardanUzaklık
imgSize = 300

folder = "Data/C"
counter = 0

labels = ["A", "B", "C"]  # datas

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img) #img'de gordugu eli tanıyıp eklemler arası dogru olusturur. hemen uzerinde img kopyalandigi icin sag-sol el farkındalıgı aktıf olmasına ragmen gozukmez.
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]  # uzunlukTanımlamaları - # bbox hand'in min ve max degerlerini koordinatta tutar.

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # beyazPencereÖlcegi

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # yalnızca elin görüntüsünü almak icin bos alanların kırpılması

        # imgCropGrey = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
        # imgCropHSV = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
        #
        # lowerValue = np.array([0, 48, 80])
        # highValue = np.array([20, 255, 255])
        #
        # colorFilterResult = cv2.inRange(imgCropHSV, lowerValue, highValue)
        # colorFilterResult = cv2.morphologyEx(colorFilterResult, cv2.MORPH_CLOSE, kernel)

        imgCropShape = imgCrop.shape  # kırpılmıs pencere

        aspectRatio = h / w  # yükseklik/Genislik (en/boy oranı)

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)  # anlik w hesaplar
            imgReSize = cv2.resize(imgCrop, (wCal, imgSize))  # imgYeniBoyut
            imgReSizeShape = imgReSize.shape
            wGap = math.ceil((imgSize - wCal) / 2)  # KoselereOlanWboslugu
            imgWhite[:, wGap:wCal + wGap] = imgReSize  # beyaz Pencere İcine Elin w'ye dayalı fazlasını Kırpıp Atma. #elLokasyonu

            prediction, index = classifier.getPrediction(imgWhite, draw=False)  # index'e dayalı imgWhite'da tahmin yapar.

            # beyaz Pencerede Genislik Oturu Tasma Durumunda App Durmaz.

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)  # anlik h hesaplar
            imgReSize = cv2.resize(imgCrop, (imgSize, hCal))  # imgYeniBoyut
            imgReSizeShape = imgReSize.shape
            hGap = math.ceil((imgSize - hCal) / 2)  # koselere Olan h boslugu
            imgWhite[hGap:hCal + hGap, :] = imgReSize  # beyaz Pencere İcine Elin h'a dayalı fazlasını Kırpıp Atma. #elLokasyonu

            prediction, index = classifier.getPrediction(imgWhite, draw=False)  # index'e dayalı imgWhite'da tahmin yapar.

            # beyaz Pencerede Yukseklik Oturu Tasma Durumunda App Durmaz.

        print(prediction, labels[index])

        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)  #eli cevreleyen mor cerveve.
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 255), 2) #cerceve üzerine index'ten yakalanan harfi cıktı verir.

        cv2.imshow("ImageCrop", imgCrop)  # kucukPencereElBoyutunaGore
        cv2.imshow("ImageWhite", imgWhite)  # beyazPencere
        # cv2.imshow("colorFilterResult", imgCropGrey)

    cv2.imshow("Camera", imgOutput)  # pencereAdıveİçerigi
    cv2.waitKey(1)  # videolaştırma
