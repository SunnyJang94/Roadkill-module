# 모듈 선언
import board
import neopixel
import time

# GPIO pin 설정 및 neopixel 선언
pixel_pin = board.D21
pixels = neopixel.NeoPixel(board.D21, 64)

is_running = True

# neopixel RGB를 모두 on시켜 white로 변환
pixels.fill((255, 255, 255))
pixels.show()
