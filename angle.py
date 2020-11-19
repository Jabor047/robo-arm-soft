from pyfirmata import Arduino, util
from time import sleep

board = Arduino('/dev/ttyUSB0')
RightLeftServoPin = board.get_pin('d:5:s')
ForwardBackwardServoLPin = board.get_pin('d:6:s')
UpDownServoPin = board.get_pin('d:7:s')
# gripServoPin = board.get_pin('d:8:p')
sleep(5)

iterSer = util.Iterator(board)
iterSer.start()

RightLeftAngle = RightLeftServoPin.read()
ForwardBackwardAngle = ForwardBackwardServoLPin.read()
UpDownServoAngle = UpDownServoPin.read()

print(RightLeftAngle)
print(ForwardBackwardAngle)
print(UpDownServoAngle)

angle = 10
RightLeftServoPin.write(angle)
UpDownServoPin.write(angle)
ForwardBackwardServoLPin.write(angle)
