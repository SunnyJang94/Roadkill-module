#include <SoftwareSerial.h>
#define HC11_receive //함수 정의


//HC11 핀 연결
int RXpin = 11;

int TXpin = 12;

int SETpin = 10;

int Chanel = 5; 




SoftwareSerial HC11(RXpin,TXpin);


void setup() {

//5개의 LED(D2~D6번핀)와 1개의 buzzer(D7번핀) 핀 설정
  for(int i = 7; i >=2; i--){
  pinMode(i, OUTPUT); 
  }
  

//시리얼 포트를 열고 1초당 전송 될 데이터의 양 설정
  Serial.begin(9600); 

  HC11.begin(9600);

  char CHANEL[4];

  sprintf(CHANEL,"%03d",Chanel);


  //CC41모듈의 채널을 설정

  pinMode(SETpin,OUTPUT);

  digitalWrite(SETpin,LOW);

  delay(500);

  HC11.print("AT+C");

  HC11.print(CHANEL);

  HC11.print("\r\n");

  delay(1000);

  while(HC11.available()>0){

    Serial.write(HC11.read());

  }

  digitalWrite(SETpin,HIGH);

}



void loop() {

// 차량용 단말기 receive 소스코드
   #ifdef HC11_receive
  
if(HC11.available() > 0) // 가로등 모듈에서 동물을 감지했다는 통신을 받으면
{ 
   for(int j = 1; j <= 6; j++){ //for문 안의 LED와 buzzer 제어 6번 반복

     tone(7, 4000, 50); //buzzer ON
      
    //5개의 LED를 왼쪽부터 차례대로 ON
    for(int i = 6; i >= 2; i--){ 
      digitalWrite(i, HIGH);
      delay(100);
      
      }

      //5개의 LED를 한꺼번에 OFF
      for(int i = 6; i >= 2; i--){
        
        digitalWrite(i, LOW); 
}

        

        // 5개의 LED를 오른쪽부터 차례대로 ON
        for(int i = 2; i <= 6; i++){
          digitalWrite(j, HIGH); delay(100);
          }

          //5개의 LED 한꺼번에 OFF
          for(int i = 6; i >= 2; i--){
            digitalWrite(i, LOW);       
            }
    }
    
      noTone(7); // buzzer OFF



  }

  #endif
}
