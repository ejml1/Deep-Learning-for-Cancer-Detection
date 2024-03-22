'''
Turn classification reports to CSV files
'''

import csv
import sys
import pickle

def save_classification_report_to_csv(class_report, filename):
    lines = class_report.split('\n')
    rows = []
    for line in lines[2:-5]: 
        row = line.strip().split()
        rows.append(row)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        writer.writerows(rows)

def main():
    class_report_loc = sys.argv[1]
    filename = sys.argv[2]

    with open(class_report_loc, 'rb') as f:
        class_report = pickle.load(f)
    
    save_classification_report_to_csv(class_report, filename)

if __name__ == "__main__":
    main()