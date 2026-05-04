import os

# Define the directory containing your files
directory = "/data/yjy_data/palette_color/sen_data_new2/train/B"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file name contains 's2'
    if '_s2' in filename:
        # Create the new file name
        new_filename = filename.replace('_s2', '_s1')
        # Get the full path of the old and new file names
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} to {new_file}')
