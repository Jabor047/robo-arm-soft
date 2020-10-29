#include <Servo.h>

int baseServoPin = 5;
int sideServoLPin = 6;
int sideServoRPin = 7;
int gripServoPin = 8;

Servo baseServo, sideServoL, sideServoR, gripServo;

void setup() {
  // put your setup code here, to run once:
  baseServo.attach(baseServoPin);
  sideServoL.attach(sideServoLPin);
  sideServoR.attach(sideServoRPin);
  gripServo.attach(gripServoPin);

  baseServo.write(0);
  delay(1000);
  baseServo.write(90);
  delay(1000);
  baseServo.write(0);
  
}

void loop() {
  // put your main code here, to run repeatedly:

}
