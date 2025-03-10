import os
import time
import paho.mqtt.client as mqtt
import polars as pl


broker = 'localhost'
port = 1883
data_topic = 'sensor/data'
ack_topic = 'sensor/ack'
batch_size = 1*20  # Number of lines to send together
client = mqtt.Client()
client.connect(broker, port, 60)


waiting_for_ack = False

# incoming messages
def on_message(client, userdata, msg):
    global waiting_for_ack
    if msg.topic == ack_topic and msg.payload.decode('utf-8') == "ACK":
        waiting_for_ack = False

client.on_message = on_message
client.subscribe(ack_topic)

client.loop_start()


def send_data():
    global waiting_for_ack
    all_files = os.listdir('data/test')
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))

    for csv_file in csv_files:
        df = pl.read_csv(f'data/test/{csv_file}', separator=';')

        total_lines = len(df)
        print(f"Processing file: {csv_file} with {total_lines} lines.")

        # Sends data in batches of 20 lines
        for idx in range(0, total_lines, batch_size):
            batch = df[idx:idx + batch_size]
            message = '\n'.join([','.join(map(str, row)) for row in batch.rows()])
            

            waiting_for_ack = True
            client.publish(data_topic, message)
            print(f'Sent batch starting at line {idx + 1} of {total_lines}')
            
            # Wait for acknowledgment
            while waiting_for_ack:
                time.sleep(0.01)  # Short sleep to prevent busy-waiting

    client.disconnect()
    print("Finished sending data.")


send_data()

client.loop_stop()