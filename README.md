# Cancer-Recurrance-DS
An alorith to predict the probability of cancer reccurance in a patient who has had breast cancer before.

FIND LINKS TO COLABS BELOW

I ran through a series of different classifications using different parameters and dimensionality reduction.
From this I found that Support Vector Machine was the best preforming algorithm which gave me the best results.

# Recurrance.py - https://colab.research.google.com/drive/19e6dfWux-mjvOjwyA41im9Tu-74XyPvo#scrollTo=TU4WYOWct23S&forceEdit=true&sandboxMode=true
The original .xls file had inconsistencies in the data. Due to this I tried various methods to work around this.
The first way I tried to get arThis is my submission for the Data Science module in the JHUB Coding Scheme.
I ran through a series of different classifications using different parameters and dimensionality reduction.
From this I found that Support Vector Machine was the best preforming algorithm which gave me the best results.

# Recurrance-reformatted.py
The original .xls file had inconsistencies in the data. Due to this I tried various methods to work around this.
The first way I tried to get around this problem was by manually editing the file and removing all data from the cells which had
a date in them. This meant I had to use a library called SimpleImputer to find the mode of the 2 columns that were affected.
This provided me with my best results, 77.5% accuracy.

I then wanted to ensure that I used the original data unedited in case that was what the markers wanted. From this I decided
that I would drop the 2 columns from the dataset and continue with further trials of each classification. This provided me
with a 75.8% accuracy, which seems like a negligible amount, but after all it is about predicting recurrence rates of cancer
so we want to be as precise as possible.

As I about to submit this I have noticed that the challenge has been slightly amended in the last few days whilst I have been 
writing this code in my spare time. I will happily reformat it to be in line with the newly requested structure if required but 
for now I am continuing on with the scheme and shall return to this if needs be.
ound this problem was by manually editing the file and removing all data from the cells which had
a date in them. This meant I had to use a library called SimpleImputer to find the mode of the 2 columns that were affected.
This provided me with my best results, 77.5% accuracy.

https://colab.research.google.com/drive/1c1PB1YafjDBGKc387RsHxRlIn_a9TJxB#scrollTo=37puETfgRzzg&forceEdit=true&sandboxMode=true
I then wanted to ensure that I used the original data unedited incase that was what the markers wanted. From this I decided
that I would drop the 2 columns from the dataset, and continue with further trials of each classification. This provided me
with a 75.8% accuracy, which seems like a negligable amount, but after all it is about predicting recurrence rates of cancer
so we want to be as precise as possible.

As I about to submit this I have noticed that the challange has been slightly ammended in the last few days whilst I have been 
writing this code in my spare time. I will happily reformat it to be in line with the newly requested structure if required but 
for now I am continuing on with the scheme and shall return to this if needs be.
