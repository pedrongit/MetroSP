import serial
import time

# Configurações da porta serial
porta = 'COM4'  # Substitua pela porta serial correta do seu Arduino
baud_rate = 9600

# Cria uma instância do objeto Serial
arduino = serial.Serial(porta, baud_rate)

# Função para ligar o relé
def ligar_rele():
    arduino.write(b'H')  # Envia o caractere 'H' para ligar o relé
    print("Relé ligado")

# Função para desligar o relé
def desligar_rele():
    arduino.write(b'L')  # Envia o caractere 'L' para desligar o relé
    print("Relé desligado")

# Aguarda a comunicação serial ser estabelecida
time.sleep(2)
desligar_rele()
# Exemplo de uso: liga e desliga o relé em intervalos de 2 segundos
for _ in range(5):
    ligar_rele()
    time.sleep(2)
    desligar_rele()
    time.sleep(2)

# Fecha a comunicação serial
arduino.close()