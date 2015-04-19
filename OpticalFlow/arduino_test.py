import serial

ser = serial.Serial('/dev/tty.usbmodem1411', 9600)
ser.write('1 1000 150\n')