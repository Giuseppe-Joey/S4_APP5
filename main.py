


# .py files to import
import FFT_Numpy_examples
import seminaire_python
import APP

# starting a timer for the app launching total time
import time




def main():

    print("Hello Lucas!")

    #FFT_Numpy_examples.generate_signal()
    #FFT_Numpy_examples.using_FFT()
    #FFT_Numpy_examples.using_FFT_scipy()
    #FFT_Numpy_examples.filtering_signal_using_FFT()

    #APP.TEST1()

    # tests with .WAV files
    note_guitare_LAd = './sounds/note_guitare_LAd.wav'
    note_basson_1000_Hz = './sounds/note_basson_plus_sinus_1000_Hz.wav'
    test_file = './sounds/test1.wav'

    print("Wave file parameters of {}".format(note_basson_1000_Hz))
    APP.wav_file_parameters(note_basson_1000_Hz)


    #APP.write_wav_file(test_file)
    #APP.analyze_wav_file(note_basson_1000_Hz)
    #APP.analyze_wav_file_2(note_basson_1000_Hz)


    #APP.sound_processing(note_guitare_LAd)
    # APP.sound_processing_test(note_basson_1000_Hz)
    APP.sound_processing_filter_1000Hz(note_basson_1000_Hz)









if __name__ == "__main__":
    # starting a counter
    app_launch_time = time.time()

    # running main()
    main()

    # showing the time the app took to launch
    print("*******************************************************************")
    print("The Script took {} seconds total".format('{:,.3f}'.format(time.time() - app_launch_time)))
    print("*******************************************************************")