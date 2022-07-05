
# Program Name :            S4_APP5
PROGRAM_VERSION = 1.0
# GitHub : 	                https://github.com/Giuseppe-Joey/S4_APP5
# Author's Name : 	        Giuseppe Lomonaco
# Author's Team :           N/A
# Author's College :        University of Sherbrooke
# Author's Field of Study : Electrical Engineering
# Author's Intern # :       N/A
# Season / Session  :       Summer 2022
# For : 	                APP5





# .py library to import
import FFT_Numpy_examples
import seminaire_python
import tests
import APP

# starting a timer for the app launching total time
import time







def main():

    notes_frequency_dictionnary = APP.notes_frequency_dictionnary()
    print("Here's the notes dictionnary: {}".format(notes_frequency_dictionnary))

    notes_K_index_dictionnary = APP.notes_K_index_dictionnary()
    print("Here's the notes dictionnary: {}".format(notes_K_index_dictionnary))

    notes_factor_dictionnary = APP.notes_factor_dictionnary()
    print("Here's the notes dictionnary: {}".format(notes_factor_dictionnary))


    #FFT_Numpy_examples.generate_signal()
    #FFT_Numpy_examples.using_FFT()
    #FFT_Numpy_examples.using_FFT_scipy()
    #FFT_Numpy_examples.filtering_signal_using_FFT()

    #tests.TEST1()

    # tests with .WAV files
    note_guitare_LAd = './sounds/note_guitare_LAd.wav'
    test_file = './sounds/test1.wav'
    note_basson_1000_Hz = './sounds/note_basson_plus_sinus_1000_Hz.wav'
    note_basson_filtered = './sounds/note_basson_filtered.wav'


    print("Wave file parameters of {}".format(note_basson_1000_Hz))
    tests.wav_file_parameters(note_basson_1000_Hz)


    #tests.write_wav_file(test_file)
    #tests.analyze_wav_file(note_basson_1000_Hz)
    #tests.analyze_wav_file_2(note_basson_1000_Hz)


    # tests.sound_processing_tests(note_guitare_LAd)
    # tests.sound_processing_test(note_basson_1000_Hz)
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