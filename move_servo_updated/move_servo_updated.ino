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
    Pos = Myservo.read();
    
    if(inByte == "left"){
      delay(1000);
      Myservo.write(Pos-90);
    }else if (inByte == "right") {
      delay(1000);
      Myservo.write(Pos+90);
    }
  }
}
