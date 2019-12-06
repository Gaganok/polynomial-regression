'''
Created on Dec 3, 2019

@author: Grand
'''

import numpy as np
import pandas as ps;

n = 800;

def main():
    data = ps.read_csv("diamonds.csv");
    
    cuts = data["cut"].unique();
    colors = data["color"].unique();
    clarities = data["clarity"].unique();
        
    cuts_ds = {};
    colors_ds = {};
    clarities_ds = {};
    
    fillDataset(data, cuts, cuts_ds, "cut");
    fillDataset(data, colors, colors_ds, "color");
    fillDataset(data, clarities, clarities_ds, "clarity");
    
    featureTargetSplit(cuts_ds);
    featureTargetSplit(colors_ds);
    featureTargetSplit(clarities_ds);
    
    countCombinatios(data, cuts, colors, clarities);
    
    
def fillDataset(df, features, ds, feature_title):
    for feature in features:
        ds[feature] = df[df[feature_title] == feature]

def featureTargetSplit(ds):
    for key in ds:
        ds[key] = (ds[key][["carat", "depth", "table"]], ds[key]["price"])
        
def countCombinatios(df, cuts, colors, clarities):
    for cut in cuts:
        for color in colors:
            for clarity in clarities:
                data = df[(df["cut"] == cut) & (df["color"] == color) & (df["clarity"] == clarity)];
                length = len(data);
                print("Cut: " + cut + " Color: " + color + " Clarity: " + clarity + " Count/Total Amount: " + str(length));
                if(length > n):
                    #use
                    print();
main();