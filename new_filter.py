import csv
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def sliding_window_prediction(rows, window_size_median, window_size_light, certainty_threshold=0.5):
    print(f"\nProcessing {len(rows)} rows with window sizes: median={window_size_median}, light={window_size_light}")
    print(f"Certainty threshold: {certainty_threshold}")
    
    predictions = []
    current_median_state = '11'
    current_light_state = '11'
    last_numeric_median = None  # Store last numeric state before 11
    last_numeric_light = None   # Store last numeric state before 11
    
    for i in range(len(rows)):
        if i % 1000 == 0:
            print(f"Processing row {i}/{len(rows)} ({(i/len(rows)*100):.1f}%)")
            
        window_median = rows[max(0, i-window_size_median+1):i+1]
        window_light = rows[max(0, i-window_size_light+1):i+1]
        
        median_predictions = [row['clf_roll_agg_median.pkl'] for row in window_median];
        light_predictions = [row['clf_roll_agg_median_light.pkl'] for row in window_light]
        
        median_counter = Counter(median_predictions)
        light_counter = Counter(light_predictions)
        
        # Process median window
        window_prediction_median, last_numeric_median = process_window(
            counter=median_counter,
            window_size=len(window_median),
            current_state=current_median_state,
            last_numeric=last_numeric_median,
            threshold=certainty_threshold
        )
        current_median_state = window_prediction_median
        
        # Process light window
        window_prediction_light, last_numeric_light = process_window(
            counter=light_counter,
            window_size=len(window_light),
            current_state=current_light_state,
            last_numeric=last_numeric_light,
            threshold=certainty_threshold
        )
        current_light_state = window_prediction_light
        
        predictions.append({
            'window_prediction_median': window_prediction_median,
            'window_prediction_light': window_prediction_light
        })
    
    print("Finished processing all rows")
    return predictions

def process_window(counter, window_size, current_state, last_numeric, threshold):
    # If current state is '11', we can go to:
    # 1. '10' if it meets threshold
    # 2. last_numeric if it meets threshold
    # 3. (last_numeric - 1) if it meets threshold
    # 4. Stay at '11'
    if current_state == '11':
        # Check for 10
        ten_count = counter.get('10', 0)
        if ten_count/window_size > threshold:
            return '10', None
        
        # If we have a last numeric state, check if we can return to it or one below
        if last_numeric is not None:
            current_value = int(last_numeric)
            current_count = counter.get(last_numeric, 0)
            if current_count/window_size > threshold:
                return last_numeric, last_numeric
            
            next_value = current_value - 1
            if next_value >= 0:
                next_count = counter.get(str(next_value), 0)
                if next_count/window_size > threshold:
                    return str(next_value), None
        
        return '11', last_numeric
    
    # If we're in a numeric state
    current_value = int(current_state)
    next_value = current_value - 1
    
    # Check if we can stay in current state
    current_value_count = counter.get(str(current_value), 0)
    if current_value_count/window_size > threshold:
        return str(current_value), None
    
    # Check if we can go to next lower value
    if next_value >= 0:
        next_value_count = counter.get(str(next_value), 0)
        if next_value_count/window_size > threshold:
            return str(next_value), None
    
    # If we need to go to 11, save the current state as last_numeric
    return '11', str(current_value)

def save_predictions_to_csv(rows, window_predictions, output_file):
    print(f"\nSaving predictions to {output_file}")
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Time', 'Actual_Class', 'clf_roll_agg_median.pkl', 'clf_roll_agg_median_light.pkl', 'windowed_median', 'windowed_light']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        
        writer.writeheader()
        rows_written = 0
        for row, pred in zip(rows, window_predictions):
            writer.writerow({
                'Time': row['Time'],
                'Actual_Class': row['Actual_Class'],
                'clf_roll_agg_median.pkl': row['clf_roll_agg_median.pkl'],
                'clf_roll_agg_median_light.pkl': row['clf_roll_agg_median_light.pkl'],
                'windowed_median': pred['window_prediction_median'],
                'windowed_light': pred['window_prediction_light']
            })
            rows_written += 1
    print(f"Successfully wrote {rows_written} rows to {output_file}")

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return accuracy, precision, recall, f1, mcc

def analyze_window_sizes(input_file, window_size_median, window_size_light, certainty_threshold=0.7):
    print(f"\nStarting analysis...")
    print(f"Reading input file: {input_file}")
    
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        rows = list(reader)
    
    print(f"Successfully loaded {len(rows)} rows from input file")
    
    window_predictions = sliding_window_prediction(rows, window_size_median, window_size_light, certainty_threshold)
    
    output_file = f'New_predictions_Roll_12Hours_12Classes_median_{window_size_median}_light_{window_size_light}.csv'
    save_predictions_to_csv(rows, window_predictions, output_file)

    print("\nCalculating metrics...")
    y_true = [row['Actual_Class'] for row in rows]
    y_pred_median = [pred['window_prediction_median'] for pred in window_predictions]
    y_pred_light = [pred['window_prediction_light'] for pred in window_predictions]

    print("\nMetrics for Median Window Predictions:")
    accuracy, precision, recall, f1, mcc = calculate_metrics(y_true, y_pred_median)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

    print("\nMetrics for Light Window Predictions:")
    accuracy, precision, recall, f1, mcc = calculate_metrics(y_true, y_pred_light)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    print("\nAnalysis complete!")

# Usage
input_file = 'predictions_Roll_12Hours_12Classes.csv'
certainty_threshold = 0.4  # 40% threshold might need testing around
window_size_light = 450   # window for sequence models: light-450    single second models: light-310 
window_size_median = 500  #                             nonLight-500                       non light-350


print("Starting program execution...")
print(f"Input file: {input_file}")
print(f"Window sizes - Median: {window_size_median}, Light: {window_size_light}")
print(f"Certainty threshold: {certainty_threshold}")

analyze_window_sizes(input_file, window_size_median, window_size_light, certainty_threshold)