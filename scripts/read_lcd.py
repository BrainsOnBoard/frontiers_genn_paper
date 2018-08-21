import csv
import cv2
import time
from subprocess import call, check_output
from tempfile import NamedTemporaryFile
cap = cv2.VideoCapture(0)

start_time = time.time()
with open("power.csv", 'wb') as csv_file, NamedTemporaryFile(suffix=".png") as tmp_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(["Time [s]", "Power [W]"])
    while True:
        frame = cap.read()
        if frame[0]:
            digit = frame[1][350:395,245:345:]
            cv2.imwrite(tmp_file.name, digit)

            output = check_output(["ssocr", "--number-digits=-1", tmp_file.name])
            power = float(output)
            print("%fW" % power)
            csv_writer.writerow([time.time() - start_time, power])
