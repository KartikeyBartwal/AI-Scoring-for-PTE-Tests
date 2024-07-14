from fastapi import FastAPI 
from pydantic import BaseModel 
from fastapi import FastAPI, File, UploadFile , status
import uvicorn 
import zipfile 
import os 
import aiofiles 
from fastapi.exceptions import HTTPException
import patoolib
import random 


'''
Zip File: audio.wav , user_written_text.txt , target_text.txt
'''
class MyClass(BaseModel):
    speech_path : str 
    user_text : str 
    target_text : str 

app = FastAPI()

# DELETES THE ZIP FILE AND ITS EXTRACTED CONTENTS
def cleanup():
   # DELETE ALL THE FILES IN THE DIRECTORY ZIP_FILE
   directory_path = "zip_files"
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")
     
   os.rmdir("zip_files")
   # DELETE THE ZIP FILE
   os.remove("zip.rar")        

def scripted_test_scoring():
    output = random.randint(0 , 100)
    # cleanup()
    return output

def unscripted_test_scoring():
    output = random.randint(0 , 100)
    # cleanup()
    return output


@app.get('/')
async def root():
    return "let us begin"



@app.post('/unscripted_task')
async def unscripted_root(audio_file : UploadFile, text_file : UploadFile , target_text_file : UploadFile):
   return unscripted_test_scoring()


@app.post('/scripted_task')
async def scripted_root(audio_file : UploadFile, text_file : UploadFile , target_text_file : UploadFile):
   return scripted_test_scoring()


# @app.post('/unscripted_task_')
# async def unscripted_root_(file: UploadFile):
#     ############################################ DOWNLOADING THE ZIP FILE ############################################
#     CHUNK_SIZE = 1024 * 1024
#     filepath = os.path.join('./', os.path.basename(file.filename))
    
#     print(f"Starting upload for file: {file.filename}")
#     print(f"Destination file path: {filepath}")
    
#     async with aiofiles.open(filepath, 'wb') as f:
#         print("Opened file in write mode")
#         while True:
#             chunk = await file.read(CHUNK_SIZE)
#             if not chunk:
#                 print("No more chunks to read, breaking the loop")
#                 break
#             await f.write(chunk)
#             print(f"Wrote a chunk of size: {len(chunk)} bytes")
    
#     await file.close()
#     print(f"Closed the file: {file.filename}")
    
#     print(f"Successfully uploaded file: {file.filename}")


#     ###################################################### EXTRACTING THE CONTENTS OF THE ZIPFILE ############################
#     '''
#     Zip File: audio.wav , user_written_text.txt , target_text.txt
#     '''
#     patoolib.extract_archive("zip.rar" , outdir= "zip_files")


#     return unscripted_test_scoring()


# @app.post('/scripted_task_')
# async def scripted_root_(file: UploadFile):
#     ############################################ DOWNLOADING THE ZIP FILE ############################################
#     CHUNK_SIZE = 1024 * 1024
#     filepath = os.path.join('./', os.path.basename(file.filename))
    
#     print(f"Starting upload for file: {file.filename}")
#     print(f"Destination file path: {filepath}")
    
#     async with aiofiles.open(filepath, 'wb') as f:
#         print("Opened file in write mode")
#         while True:
#             chunk = await file.read(CHUNK_SIZE)
#             if not chunk:
#                 print("No more chunks to read, breaking the loop")
#                 break
#             await f.write(chunk)
#             print(f"Wrote a chunk of size: {len(chunk)} bytes")
    
#     await file.close()
#     print(f"Closed the file: {file.filename}")
    
#     print(f"Successfully uploaded file: {file.filename}")


#     ###################################################### EXTRACTING THE CONTENTS OF THE ZIPFILE ############################
#     '''
#     Zip File: audio.wav , user_written_text.txt , target_text.txt
#     '''
#     patoolib.extract_archive("zip.rar" , outdir= "zip_files")

#     return scripted_test_scoring()
