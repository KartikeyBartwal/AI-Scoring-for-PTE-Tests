from pydub import AudioSegment

def convert_opus_to_mp3(input_file, output_file):
    # LOAD OPUS FILE
    audio = AudioSegment.from_file(input_file, format="opus")

    # EXPORT AS MP3 FILE
    audio.export(output_file, format="mp3")
    print(f"Converted '{input_file}' to '{output_file}' successfully.")


convert_opus_to_mp3("input_file.opus", "output_file.mp3")
