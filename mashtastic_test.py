import meshtastic
import meshtastic.serial_interface
from pubsub import pub
import time

def on_receive(packet, interface):
    print("Received packet:", packet)

def on_connection(interface, topic=pub.AUTO_TOPIC):
    print("Device connected")
    interface.sendText("Single node test message")

pub.subscribe(on_receive, "meshtastic.receive")
pub.subscribe(on_connection, "meshtastic.connection.established")

print("Connecting to device...")
interface = meshtastic.serial_interface.SerialInterface()

# Keep running and listening
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    interface.close()
