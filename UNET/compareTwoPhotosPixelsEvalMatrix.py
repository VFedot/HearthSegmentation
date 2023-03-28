import csv
import os
from PIL import Image

# Define the folders where the images are stored
my_folder = 'MyMask'
ai_folder = 'AiMask'

# Iterate over all the files in the folders
with open('output.csv', mode='w', newline='') as file:
    # Define the CSV writer
    writer = csv.writer(file)

    # Write the headers
    writer.writerow(['Filename',
                     'White pixel common|True positive', 'Black pixel common|True negative',
                     'Black pixel differences|False positives', 'White pixel differences|False negatives',
                     'White pixel common in percent|True positives', 'Black pixel common in percent|True negative',
                     'Black pixel differences in percent|False positives', 'White pixel differences in percent|False negative'])

    # Iterate over all the files in the folders
    for filename in os.listdir(my_folder):
        if filename.endswith('.png'):
            # Load the images
            my_mask = Image.open(os.path.join(my_folder, filename)).convert('1')  # convert to binary
            ai_mask = Image.open(os.path.join(ai_folder, filename)).convert('1')  # convert to binary

            # Get the size of the images
            width, height = my_mask.size

            # Initialize counters
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            # # Compare pixel values
            # for x in range(width):
            #     for y in range(height):
            #         my_pixel = my_mask.getpixel((x, y))
            #         ai_pixel = ai_mask.getpixel((x, y))
            #         if my_pixel != ai_pixel:
            #             if my_pixel == 0:
            #                 false_positives += 1
            #             else:
            #                 false_negatives += 1

            # Compare pixel values
            for x in range(width):
                for y in range(height):
                    my_pixel = my_mask.getpixel((x, y))
                    ai_pixel = ai_mask.getpixel((x, y))
                    if my_pixel == ai_pixel:
                        if my_pixel == 255 and ai_pixel == 255:
                            true_positive +=1
                        else:
                            true_negative +=1
                    else:
                        if my_pixel == 0 and ai_pixel == 255:
                            false_positive += 1
                        else:
                            false_negative += 1

            # Calculate percentage of differences
            total_pixels = width * height
            # true_positive = total_pixels - false_positives
            # true_negatives = total_pixels - false_negatives
            #
            true_positive_percent = (true_positive/(true_positive+false_negative)) * 100
            true_negatives_percent = (true_negative/(true_negative+false_positive)) * 100

            false_positives_percent = (false_positive/(false_positive+true_negative)) * 100
            false_negatives_percent = (false_negative/(false_negative+true_positive)) * 100

            # Write the data to the CSV file
            writer.writerow([filename, true_positive, true_negative, false_positive, false_negative, true_positive_percent, true_negatives_percent, false_positives_percent, false_negatives_percent])
