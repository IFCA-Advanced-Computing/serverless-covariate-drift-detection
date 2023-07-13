# main-service.py for execute EMC (Embedding Matrix Generator) service.
# The main idea of this service is to run and list the output bucket of the DRS service to see when N embedding has been done, if the N results are not there, the service does  nothing else. Once you have N embedding create an array (json object) with the results of the DRS service.


import os
import sys
import time
from minio import Minio
import urllib3
import io
import ast
import json

#Function to create object with data in json format
class format:
    def __init__(self, alpha, value,return_input_values):
        self.alpha = alpha
        self.values = value
        self.return_input_values= return_input_values


print("Running service ...")

#Creation of the minio object that will connect to the minIO server
client = Minio(
    "minio.frosty-grothendieck5.im.grycap.net",
    access_key="minio",
    secret_key="minio123", 
    secure=True,
    region="us-east-1",
    
)
# Number of objects (N embedding)
objCount=890

# List the output bucket of the DRS service
objects = client.list_objects("drs", recursive=True, prefix="output/",)
objList=0

#Count the number of items in the bucket
for obj in objects:
    objList=objList+1

#Determine if there is N embedding in the bucket
if objList>objCount+1:
    dataFiles = []
    objects = client.list_objects("drs", recursive=True, prefix="output/",)
    z=0
    for i in objects:
            print(i)
            #Read object content inside bucket
            response = client.get_object(
                bucket_name = 'drs',
                object_name= i.object_name
                )

            #Convert to string format ('utf-8')
            x=str(response.read(),'utf-8')
            print(x)
            #Determine if the object has content (eliminate problem with remote MinIO server
            #that leaves a blank object on the first run)

            if x!="":
                #Adapt the data to the input format for the DDS service and create an array of values
                x1=x.strip('\n')
                b = ast.literal_eval(x1)
                dataFiles.append(b)
                z=0
            else:
                z=1
    #In case there is no blank object
    if z==0:
        #Create json object with data for DDS service
        values=format(float('0.05'),dataFiles,True)
        data = json.dumps(values.__dict__)
        
        #Write the file with the N embedding data in the service output bucket
        out=sys.argv[2]
        f = open(str(out)+".json", "w")
        f.write(str(data))
        f.close()
        
        #Delete the objects in the input bucket (bucket drs/output) to be able to start the next N embedding collection
        objects = client.list_objects("drs", recursive=True, prefix="output/",)
        for ii in objects:
             client.remove_object("drs", ii.object_name)
             time.sleep(1) # bucket load time
        
        print("Process completed ...")
else:
    print("There are not enough objects in the bucket  ")
