
import time
import pandas as pd
from datetime import datetime
import csv
from math import sin, cos, sqrt, atan2, radians
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

## Funtion to calculate distance between gps coordinates
def metre_distance(lat1, lon1, lat2, lon2):
    approx_earth_radius = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = approx_earth_radius * c

# convert uuid into from alphanumeric to integer
def uuid_to_int(uuid):
    val = 0
    for i in range(len(uuid)):
        val += (10**i)*ord(uuid[i])
    return val

# convert time from csv timestamp form to integer
def data_to_timestamp(csv_ts):
    if len(csv_ts) > 18:
        dt = datetime(int(csv_ts[0:4]), int(csv_ts[5:7]), int(csv_ts[8:10]),
                  int(csv_ts[11:13]), int(csv_ts[14:16]), int(csv_ts[17:19]))
    elif len(csv_ts) == 18:
        dt = datetime(int(csv_ts[0:4]), int(csv_ts[5:7]), int(csv_ts[8:10]),
                  int(csv_ts[11:12]), int(csv_ts[13:15]), int(csv_ts[16:18]))
    else:
        print('bad time type')
        dt = datetime(0, 0, 0, 0, 0, 0)

    #print(dt)
    #print(datetime.timestamp(dt))
    return int(datetime.timestamp(dt))


# check that interne users are unique
def unique_users_check(set1, set2):
    plen = len(set1)
    print('before: ' + str(len(set1)))
    for val in set2:
        set1.add(val)
    print('after: ' + str(len(set1)))

    if plen == len(set1):
        print('interne users are unique and the same in both')
    else:
        print('interne users are not unique')


# tid_to_ind should map trip id's to the index of their label
# 1.) check if trip id is in map, if not set this index as a stop
# 2.) if trip id IS already in the map, change previous index from stop to in_trip
# and overwrite trip_id mapping from previous index to current index
def get_stops_using_trip_id_and_time(interne_tid, interne_time, tid_to_ind,
                                     tid_to_ind_list, interne_output):
    for i in range(len(interne_tid)):
        if (interne_tid[i] in tid_to_ind):
            tid_to_ind_list[interne_tid[i]].append(i)
            if (interne_time[i] > interne_time[tid_to_ind[interne_tid[i]]]):
                interne_output[ tid_to_ind[interne_tid[i]] ] = 0
                tid_to_ind[interne_tid[i]] = i
                interne_output.append(1)
            else:
                interne_output.append(0)
        else:
            interne_output.append(1)
            tid_to_ind[interne_tid[i]] = i
            tid_to_ind_list[interne_tid[i]] = [i]


# get train/test datasets
def get_train_and_test_indexes(train, test, interne_tid, tmask):
    l = len(interne_tid) # get length of datasets
    tlen = 0 # length of training dataset
    for k in tid_to_ind_list:
        if (tlen <= 0.78*l):
            tlen += len(tid_to_ind_list[k])
            for i in tid_to_ind_list[k]:
                train.append(i)
                tmask.append(True)
        else:
            for i in tid_to_ind_list[k]:
                test.append(i)
                tmask.append(False)

##def test_def(f, a = 4, b = 8):
##    print('the value of a : ', a)

def convert_uuid_time_to_numbers(uuid, time, pt, lat, lon, speed, tid = 0):
    # convert uuid and time to number form
    for i in range(len(uuid)):
        uuid[i] = uuid_to_int(uuid[i])
    for i in range(len(time)):
        time[i] = data_to_timestamp(time[i])

    # convert to list of numbers
    pt = list(map(int, pt))
    if tid != 0:
        tid = list(map(int, tid))
    lat = list(map(float, lat))
    lon = list(map(float, lon))
    speed = list(map(float, speed))


    
#####################################################################################
################################## PROGRAM START ####################################
start = time.time()

interne_ids = []
interne_uuid = []
interne_lon = []
interne_lat = []
interne_speed = []
interne_pt = [] # pt is point type
interne_tid = [] # tid is trip id
interne_time = []
interne_output = []
tmask = []


raw_ids = []
raw_uuid = []
raw_lon = []
raw_lat = []
raw_speed = []
raw_pt = [] # pt is point type
raw_time = []
raw_output = []

tid_to_ind = {} # basicinterney using a dictionary as a map for trip_id to index
tid_to_ind_list = {}

soauu1 = set()
soauu2 = set()

total_trips_stops = 0;

##print('opening full trips csv')
##with open('trips.csv', 'r', encoding='latin-1') as csvFile:
##    coord_read = list(csv.reader(csvFile))
##    row_len = len(coord_read)
##    total_trips_stops = row_len-1
##    print(row_len-1)
##    col_len = len(coord_read[0])
##    print(col_len-1)

print('opening interne coords 20u')

interne_csv = pd.read_csv('interne_coordinates-20_users.csv')
print(len(interne_csv[['uuid']]))
interne_feats = interne_csv[['uuid','latitude','longitude','speed',\
                             'point_type','trip_id','timestamp_txt']]
interne_feats = np.asanyarray(interne_feats)
interne_feats[:,1:3] = np.around(interne_feats[:,1:3].astype(np.double), 5)


##print('time: ', interne_feats[0][-1], 'len: ', len(interne_feats[0][-1]))
##print('time: ', interne_feats[0][-1][0:19], 'len: ', len(interne_feats[0][-1][0:19]))


## convert uuid lists into number form and time lists to timestamp form
for i in range(len(interne_feats[:,0])):
    #interne_feats[i,0] = uuid_to_int(interne_feats[i,0])
    interne_feats[i,-1] = data_to_timestamp(interne_feats[i,-1])

# check for unique users in both lists
#unique_users_check(soauu1, soauu2)

# label points stop (1) or in_trip (0)
get_stops_using_trip_id_and_time(interne_feats[:,-2], interne_feats[:,-1], tid_to_ind,
                                 tid_to_ind_list, interne_output)

interne_output = np.asanyarray(interne_output)
print('number of stops in: ', np.sum(interne_output > 0, axis=0))

# split into train and test indexes (get mask)
train = []
test = []

get_train_and_test_indexes(train, test, interne_feats[:,-2], tmask)
tmask = np.asanyarray(tmask)
            
#print('80% of dataset:', 0.8*len(interne_tid))
print('80% of dataset:', 0.8*len(interne_feats[:,-2]))
print('lenght of train set:', len(train))
print('lenght of test set:', len(test))
print('actual %s of train and test: ', 100*(len(train)/len(interne_feats[:,-2])),
      100*(len(test)/len(interne_feats[:,-2])))

##################################################### train model

interne_feats = interne_feats[:,[1,2,3,4,6]] # remove trip id from training & predictions

train_x = interne_feats[tmask]
interne_output = np.asanyarray(interne_output)
train_y = interne_output[tmask]

print('interne training sets derived')
print(len(train_x[:,0]), len(train_x[0,:]))

rndforclf = RandomForestClassifier(n_estimators=28, warm_start=False,
                                   oob_score=True, #max_features=None,
                                   random_state=0, n_jobs=-1)
rndforclf.fit(train_x, train_y)


##################################################### predict and validate

# predicting on test set

test_x = interne_feats[~tmask]
test_y = interne_output[~tmask]
test_y_hat = rndforclf.predict(test_x)
correct = 0
for i in range(len(test_x)):
    if test_y[i] == test_y_hat[i]:
        correct += 1
        #print(test_y[i], test_y_hat[i], 'true')
##    else:
##        print(test_y[i], test_y_hat[i], 'false')
print('number of stops in interne test csv: ', np.sum(test_y > 0, axis=0))
print('number of stops predicted in interne test csv: ', np.sum(test_y_hat > 0, axis=0))
print('interne clasification accuracy: ', 100*(correct/len(test_x)), '%')

print('number of stops in all interne: ', np.sum(rndforclf.predict(interne_feats) > 0, axis=0))



##################################################################### predicting on raw csv

##print('opening all raw coords csv')
##raw_csv = pd.read_csv('raw_coordinates.csv')
##raw_feats = raw_csv[['uuid','latitude','longitude','speed',\
##                     'point_type','timestamp']]
##raw_feats = np.asanyarray(raw_feats)
##
##
##for i in range(len(raw_feats[:,0])):
##    #raw_feats[i,0] = uuid_to_int(raw_feats[i,0])
##    raw_feats[i,-1] = data_to_timestamp(raw_feats[i,-1])
##
##raw_x = raw_feats[:,[1,2,3,4,5]] # remove point type from predictions
###raw_x = raw_feats
##print(len(raw_x[:,0]), len(raw_x[0,:]))
##raw_y = rndforclf.predict(raw_x)
##print('finished raw prediction')
##print('number of stops in full raw csv: ', np.sum(raw_y > 0, axis=0))
##print('raw clasification accuracy: ', 'unavailable atm')



##################################################################### predicting on full interne csv

print('opening all interne coords csv')
icoord_csv = pd.read_csv('interne_coordinates.csv')
icoord_feats = icoord_csv[['uuid','latitude','longitude','speed',\
                     'point_type','trip_id','timestamp_txt']]
icoord_feats = np.asanyarray(icoord_feats)
icoord_feats[:,1:3] = np.around(icoord_feats[:,1:3].astype(np.double), 5)

all_in_output = []
tid_to_ind1 = {} 
tid_to_ind_list1 = {}
# label points stop (1) or in_trip (0)
get_stops_using_trip_id_and_time(icoord_feats[:,-2], icoord_feats[:,-1], tid_to_ind1,
                                 tid_to_ind_list1, all_in_output)

all_in_output = np.asanyarray(all_in_output)
print('number of stops: ', np.sum(all_in_output > 0, axis=0))


for i in range(len(icoord_feats[:,0])):
    #icoord_feats[i,0] = uuid_to_int(icoord_feats[i,0])
    icoord_feats[i,-1] = data_to_timestamp(icoord_feats[i,-1])

icoord_x = icoord_feats[:,[1,2,3,4,6]] # remove trip id from training & predictions
print(len(icoord_x[:,0]), len(icoord_x[0,:]))
icoord_y = rndforclf.predict(icoord_x)
print('finished interne coord prediction')

correct = 0
for i in range(len(icoord_x)):
    if icoord_y[i] == all_in_output[i]:
        correct += 1
print('number of stops in full inerne coord csv: ', np.sum(icoord_y > 0, axis=0))
print('full interne coord clasification accuracy: ', 100*(correct/len(icoord_x)), '%')


trips_csv = pd.read_csv('trips.csv', encoding='latin-1')
trips_feats = trips_csv[['uuid','timestamp_end_txt']]
print(len(trips_csv[['uuid']]))
#trips_feats = np.asanyarray(trips_feats)


################################ PROGRAM END ###################################
end = time.time()
print("total time run: " + str(end - start))


### print datetime test
##for i in range(2):
##    dt = datetime(int(interne_time[i][0:4]), int(interne_time[i][5:7]), int(interne_time[i][8:10]),
##                  int(interne_time[i][11:13]), int(interne_time[i][14:16]), int(interne_time[i][17:19]))
##    print(dt)

