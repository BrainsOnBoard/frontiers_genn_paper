import csv
import cv2
import time
from subprocess import call, check_output, CalledProcessError
from tempfile import NamedTemporaryFile
cap = cv2.VideoCapture(0)

start_time = time.time()
with open("power.csv", 'wb') as csv_file, NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(["Time [s]", "Power [W]"])
    while True:
        frame = cap.read()
        if frame[0]:
            digit = frame[1][350:395,245:340:]
            cv2.imwrite(tmp_file.name, digit)

            try:
                output = check_output(["ssocr", "--charset=decimal", "--number-digits=-1", tmp_file.name])
                if output.count(".") == 1 and output.count("-") == 0:
                    power = float(output)
                    print("%fW" % power)
                    csv_writer.writerow([time.time() - start_time, power])
                else:
                    print("Parser output invalid: '%s'" % output)
            except CalledProcessError as err:
                if err.returncode == 2:
                    print("Unable to recognise digit '%s'" % err.output)
                    pass
                else:
                    print err.returncode
                    raise
