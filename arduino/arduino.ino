// define o pino do relé 

#define RELAY_PIN 2 

void setup() { 

 // inicializa a comunicação serial 

Serial.begin(9600); 

// configura o pino do relé como saída 

pinMode(RELAY_PIN, OUTPUT); 

// inicializa o relé em estado desligado 

digitalWrite(RELAY_PIN, LOW); 

} 

 

void loop() { 

// se houver dados disponíveis na porta serial 

if (Serial.available()) { 

     

// lê o próximo byte da porta serial 

char c = Serial.read(); 

     

// se o byte for '1', liga o relé 

if (c == '1') { 

digitalWrite(RELAY_PIN, HIGH); 

} 

     

// se o byte for '0', desliga o relé 

else if (c == '0') { 

digitalWrite(RELAY_PIN, LOW); 

} 

} 

} 