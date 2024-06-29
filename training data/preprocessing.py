from pydub import AudioSegment
import os
import warnings 

warnings.filterwarnings("ignore")

def convert_m4a_to_wav(input_file):
    print(f"Starting conversion of {input_file} to .wav")

    # Check if the input file is .m4a
    if not input_file.endswith('.m4a'):
        print(f"Error: {input_file} is not a .m4a file.")
        return
    
    # Get the directory and filename
    directory = os.path.dirname(input_file)

    print("here after the directory")
    filename = os.path.basename(input_file)
    print("here after filename")
    
    # Convert .m4a file to .wav
    try:
        track = AudioSegment.from_file(input_file, '.m4a')
        print("started with tracking")

        # Replace .m4a with .wav in the filename
        wav_filename = filename.replace('.m4a', '.wav')
        wav_path = os.path.join(directory, wav_filename)
        
        print(f'Converting: {input_file} to {wav_path}')
        track.export(wav_path, format='wav')
        
        print(f'Successfully converted: {input_file} to {wav_path}')
        return wav_path
    
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return None
    
if __name__ == "__main__":
    convert_m4a_to_wav("audio_0a2fb866-25a3-42e0-b672-a02f6b8ddd1a.m4a")