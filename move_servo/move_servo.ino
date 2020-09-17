#include <Servo.h>

Servo Myservo;
String inByte;
int Pos;

void setup(){
  Myservo.attach(9);
  Serial.begin(9600);
}

void loop() {
  if(Serial.available()){
    inByte = Serial.readString();
    Pos = inByte.toInt();
    Myservo.write(Pos);
  }
}
