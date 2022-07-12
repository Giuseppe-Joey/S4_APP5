# File Name :               main.py

# GitHub : 	                https://github.com/Giuseppe-Joey/S4_APP5
# Author's Name : 	        Giuseppe Lomonaco
# Author's Team :           N/A
# Author's College :        University of Sherbrooke
# Author's Study :          Electrical Engineering
# Author's Intern # :       N/A
# Season / Session  :       Summer 2022
# For : 	                APP5


# custom functions to import
import APP_Final

# starting a timer for the app launching total time
import time
import soundfile as sf


note_guitare_LAd = './sounds/note_guitare_LAd.wav'
note_basson_1000_Hz = './sounds/note_basson_plus_sinus_1000_Hz.wav'
note_basson_filtered = './filtered_sounds/note_basson_filtered.wav'


def main():

    # ///////////////////////////////////////////////////////////
    #                     GUITARE SECTION
    # ///////////////////////////////////////////////////////////
    signal_guitare, Fs_guitare = sf.read(note_guitare_LAd)
    APP_Final.sound_details(signal_guitare, Fs_guitare)

    k = APP_Final.find_k()
    new_signal = APP_Final.env_temp(signal_guitare, k)
    # ///////////////////////////////////////////////////////////





    # ///////////////////////////////////////////////////////////
    #                      BASSON SECTION
    # ///////////////////////////////////////////////////////////
    signal_basson, Fs_basson = sf.read(note_basson_1000_Hz)
    APP_Final.sound_details(signal_basson, Fs_basson)
    APP_Final.sound_processing_filter_1000Hz(note_basson_1000_Hz, note_basson_filtered)

    k = APP_Final.find_k()
    new_signal = APP_Final.env_temp(signal_basson, k)
    # ///////////////////////////////////////////////////////////



    # # writing back the signal into .wav file
    # test_file_filtering = './filtered_sounds/note_basson_filtered_test_lucas.wav'
    # sf.write(test_file_filtering, new_signal, samplerate=Fs)






if __name__ == "__main__":

    # starting a counter
    app_launch_time = time.time()

    # running main()
    main()

    # showing the time the app took to launch
    print("*******************************************************************")
    print("The Script took {} seconds total".format('{:,.3f}'.format(time.time() - app_launch_time)))
    print("*******************************************************************")