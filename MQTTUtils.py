import paho.mqtt.client as mqtt

class MQTTClient:
    def __init__(self, broker_address, port, client_id):
        self.broker_address = broker_address
        self.port = port
        self.client_id = client_id
        self.connected = False
        self.client = mqtt.Client(client_id)

        # Set up the callback functions
        self.client.on_connect = self.on_connect

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
            self.connected = True
        else:
            print(f"Failed to connect, return code {rc}")

    def connect(self):
        # Connect to the MQTT broker
        self.client.connect(self.broker_address, port=self.port, keepalive=60)
        # Start the loop to handle network communication, callbacks, and reconnections
        self.client.loop_start()

    def disconnect(self):
        # Stop the network loop
        self.client.loop_stop()
        # Disconnect from the broker
        self.client.disconnect()
        self.connected = False

    def publish(self, topic, message):
        if not self.connected:
            print("Not connected to MQTT broker. Cannot publish.")
            return

        # Publish a message
        self.client.publish(topic, message)
