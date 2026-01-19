import csv

# Input and output file details (naming)
input_file = "./creditcard.csv"  # Replace with your CSV file's path
output_file_prefix = "credit_card"  # Prefix for output files
records_per_file = 80000  # Number of records per file

def split_csv(input_file, output_file_prefix, records_per_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        headers = next(reader)  # Extract headers
        records = list(reader)  # Load all records

        total_records = len(records)
        num_files = (total_records // records_per_file) + (1 if total_records % records_per_file != 0 else 0)
        
        for i in range(num_files):
            start_index = i * records_per_file
            end_index = start_index + records_per_file
            chunk = records[start_index:end_index]
            output_file = f"{output_file_prefix}_{i+1}.csv"
            
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(headers)  # Write headers to the file
                writer.writerows(chunk)  # Write chunk of data
                
            print(f"Created file: {output_file} with records {start_index + 1} to {min(end_index, total_records)}")

# Run the function
split_csv(input_file, output_file_prefix, records_per_file)