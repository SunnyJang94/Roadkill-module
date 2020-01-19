# 모듈 선언
import time
import serial

# 시리얼 통신을 위한 포트 설정 및 baudrate 설정
ser = serial.Serial(
    port ='/dev/ttyUSB0',
    baudrate = 9600,
    parity =serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
    )
# 차량용 단말기에 신호 송신
ser.write(str.encode('Detect!\n'))