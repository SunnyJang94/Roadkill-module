# 모듈 선언
import board
import neopixel
import time

# GPIO pin 설정 및 neopixel 선언
pixel_pin = board.D21
pixels = neopixel.NeoPixel(board.D21, 64)

is_running = True

# neopixel RGB 모두를 off
pixels.fill((0, 0, 0))
pixels.show()