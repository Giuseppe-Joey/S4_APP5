# File Name :               main.py

# GitHub : 	                https://github.com/Giuseppe-Joey/S4_APP5
# Author's Name : 	        Giuseppe Lomonaco
# Author's Team :           N/A
# Author's College :        University of Sherbrooke
# Author's Study :          Electrical Engineering
# Author's Intern # :       N/A
# Season / Session  :       Summer 2022
# For : 	                APP5





# .py library to import
#import FFT_Numpy_examples
#import seminaire_python
#import tests
import APP
import APP_Final

# starting a timer for the app launching total time
import time









def main():


    # tests with .WAV files
    note_guitare_LAd = './sounds/note_guitare_LAd.wav'
    note_basson_1000_Hz = './sounds/note_basson_plus_sinus_1000_Hz.wav'
    note_basson_filtered = './filtered_sounds/note_basson_filtered.wav'

    print("----- Processing sound --------")
    APP.sound_processing_filter_1000Hz(note_basson_1000_Hz, note_basson_filtered)





if __name__ == "__main__":

    # starting a counter
    app_launch_time = time.time()

    # running main()
    main()

    # showing the time the app took to launch
    print("*******************************************************************")
    print("The Script took {} seconds total".format('{:,.3f}'.format(time.time() - app_launch_time)))
    print("*******************************************************************")