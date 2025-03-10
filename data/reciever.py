import os
import paho.mqtt.client as mqtt
import numpy as np
import polars as pl
import pickle
import warnings
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import (accuracy_score, matthews_corrcoef, f1_score, 
                             recall_score, precision_score, confusion_matrix)


warnings.filterwarnings("ignore")

broker = 'localhost'
port = 1883
data_topic = 'sensor/data'
ack_topic = 'sensor/ack'

output_dir = 'data/temporary'
os.makedirs(output_dir, exist_ok=True)

predictions_csv = os.path.join(output_dir, 'cabra_predictions_12Hours_12Classes.csv')

all_files = os.listdir('models/pickle')
pkl_files = list(filter(lambda f: f.endswith('.pkl'), all_files))
model_files = list(filter(lambda f: f.find('roll') < 0, pkl_files))

models = []
for model_file in model_files:
    with open(f'models/pickle/{model_file}', 'rb') as f:
        clf = pickle.load(f)
        models.append((model_file, clf))


true_labels = []
predicted_labels = {name: [] for name, _ in models}

def minmax_scale_custom(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def process_message(lines):
    global true_labels, predicted_labels
    
    df = pl.DataFrame(
        [line.split(',') for line in lines], 
        schema=['Time', 'Acc_X (mg)', 'Acc_Y (mg)', 'Acc_Z (mg)', 'Temperature (C)', 'Class']
    )
    
    df = df.with_columns([ 
        pl.col('Time').cast(pl.Int64),
        pl.col('Acc_X (mg)').cast(pl.Int32),
        pl.col('Acc_Y (mg)').cast(pl.Int32),
        pl.col('Acc_Z (mg)').cast(pl.Int32),
        pl.col('Temperature (C)').cast(pl.Float32),
        pl.col('Class').cast(pl.Int32)
    ])
    df = df.with_columns([pl.from_epoch('Time', time_unit='ms')])

    # Filter invalid reads
    df = df.filter((pl.col('Acc_X (mg)') > -5000) & (pl.col('Acc_X (mg)') < 5000))
    df = df.filter((pl.col('Acc_Y (mg)') > -5000) & (pl.col('Acc_Y (mg)') < 5000))
    df = df.filter((pl.col('Acc_Z (mg)') > -5000) & (pl.col('Acc_Z (mg)') < 5000))


    df = df.with_columns(
        pl.col('Acc_X (mg)').map_batches(lambda x: pl.Series(minmax_scale_custom(x, -3890, 4360))),
        pl.col('Acc_Y (mg)').map_batches(lambda x: pl.Series(minmax_scale_custom(x, -3110, 4440))),
        pl.col('Acc_Z (mg)').map_batches(lambda x: pl.Series(minmax_scale_custom(x, -4980, 2320))),
    )

  

    # Resample the data
    df_resample = df.select([
        pl.col('Time').median(),
        pl.col('Acc_X (mg)').median(),
        pl.col('Acc_Y (mg)').median(),
        pl.col('Acc_Z (mg)').median(),
        pl.col('Temperature (C)').median(),
        pl.col('Class').mode().first()
    ])

    #df_resample = df_resample.with_columns(
    #    Class=pl.col('Class').replace([-1], [12])
    #)

    X = df_resample.drop(['Class', 'Time']).to_numpy()
    y = df_resample['Class'].to_numpy()

    true_labels.extend(y)
    predictions = {}

    for name, clf in models:
        y_pred = clf.predict(X)
        predicted_labels[name].extend(y_pred)
        predictions[name] = y_pred[0] 

    with open(predictions_csv, 'a') as pred_file:
        if os.path.getsize(predictions_csv) == 0:
            model_names = ";".join([name for name, _ in models])
            pred_file.write(f"Time;Actual_Class;{model_names}\n")
        
        time = df_resample['Time'][0]
        actual_class = y[0]
        pred_values = [str(predictions[name]) for name, _ in models]
        row = [str(time), str(actual_class)] + pred_values
        pred_file.write(";".join(row) + '\n')

    print(f"Processed 1 second of data ({len(lines)} lines)")

    # print("\nMetrics after processing:")
    # for name, _ in models:
    #     y_pred = predicted_labels[name]
    #     acc = accuracy_score(true_labels, y_pred)
    #     mcc = matthews_corrcoef(true_labels, y_pred)
    #     f1 = f1_score(true_labels, y_pred, average='weighted')
    #     recall = recall_score(true_labels, y_pred, average='weighted')
    #     precision = precision_score(true_labels, y_pred, average='weighted')
    #     print(f'{name:<22} Acc: {acc:.2f}, Prec: {precision:.2f}, Rec: {recall:.2f}, F1: {f1:.2f}, MCC: {mcc:.2f}')



def on_message(client, userdata, msg):
    if msg.topic == data_topic:
        message = msg.payload.decode('utf-8')
        lines = message.strip().split('\n')
        
        print(f"Received message with {len(lines)} lines")
        process_message(lines)
    
        client.publish(ack_topic, "ACK")

client = mqtt.Client(protocol=mqtt.MQTTv5)  
client.on_message = on_message
client.connect(broker, port, 60)
client.subscribe(data_topic)
client.subscribe(ack_topic)

print("Starting MQTT client loop...")
client.loop_forever()