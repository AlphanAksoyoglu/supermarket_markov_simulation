import pandas as pd
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from random import shuffle

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from supermarket_visualization.visualization_constants import VC
from supermarket_visualization.visualization_library import *

if (__name__=='__main__'):
    mask, mask_checkout, customer_tile, market, background, frame  = initialize_market_layout(path_to_tiles='./supermarket_visualization/tiles.png')

    df = pd.read_csv('../data/sim_c10_till_20.csv',dtype={'customer_id':'str'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time'] = df['timestamp'].dt.strftime('%H:%M').str.lstrip('0')

    date = datetime.now()
    date = datetime(date.year, date.month, date.day, hour=df['timestamp'][0].hour, minute=df['timestamp'][0].minute)
    customers = []
    data_collector=DataCollector()
    grocery_locations, checkout_locations, entry_locations, start_locations, exit_locations = make_locations()
    # image_counter=0
    for times in list(df['time'].unique()): 
        grocery_locations, checkout_locations, entry_locations, start_locations, exit_locations = make_locations()
        df_time = df[df['time'] == date.strftime('%H:%M').lstrip('0')]
        date_text = date.strftime('%H:%M').lstrip('0')
        customer_dict = df_time[['customer_id','customer_location']].set_index('customer_id',drop=True).to_dict()['customer_location']
        #remove exiting customers
        customers = [i for i in customers if i.current_location != 'exit']
        #update next locations for customers
        for customer in customers:
            if (customer.current_location == 'checkout'):
                customer.next_location='exit'
            else:
                customer.next_location = customer_dict[f'{customer.id}']
        #insert new customers
        for customer_id,location in customer_dict.items():
            if location == 'entry':
                customers.append(Customer(customer_id,entry_locations.pop(),customer_tile))
        print([(customer.id,customer.next_location) for customer in customers])
        
        
        ###MOVE CUSTOMERS HERE
        for customer in customers:
            customer.find_path(grocery_locations,checkout_locations,exit_locations,mask,mask_checkout)
        
        path_lengths = [len(customer.path) for customer in customers]
        max_path = max(path_lengths)

        for i in range(max_path):
            for step in range(1,VC.STEPS+1):
                #print(step)
                frame = background.copy()
                market.draw(frame)
                for customer in customers:
                    customer.move(step,frame)
                #time.sleep(0.05)
                
                if ((i==max_path-1) & (step == VC.STEPS)):
                    for customer in customers:
                        customer.shop(data_collector)
                # Write some Text
                write_data_on_frame(frame,date_text,data_collector)


                cv2.imshow("frame", frame) # the cv2.imshow() method is what’s actually displaying each frame on the screen
                #cv2.imwrite(f'./images/image_'+f'{image_counter}'.zfill(4) + '.png',frame)
                #image_counter+=1
                key = chr(cv2.waitKey(1) & 0xFF)
                if key == "q":
                    break
        
                
        date += timedelta(minutes=1)        