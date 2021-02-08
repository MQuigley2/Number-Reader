#!/bin/env python
import tkinter as tk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from imageConverter import imageConverter


#Loading Pre-Trained model for digit classification
digitModel= load_model('models/trainedDigitModel')


#Function initializes Coordinates list with mouse coordinates when canvas is clicked 
def locate_xy(event):
    global Coordinates
    Coordinates=[event.x,event.y]


#Function appends new mouse coordinates to list as mouse moves and creates a smooth line between points
#Only retains 3 most recent mouse location to prevent slowdown from creating too many objects
def draw(event):
    global Coordinates
    Coordinates.append(event.x)
    Coordinates.append(event.y)
    canvas.create_line((Coordinates),width=thicknessScale.get()*3,fill=chooseColor.get().lower(),smooth=True)
    while len(Coordinates)>6:
        Coordinates.pop(0)


#defining function to clear screen
#Function needs to accept an acgument for it to be bound to key
def clearScreen(event=0):
    canvas.delete("all")
    resultString.set("Draw a Number between 0 and 9")


#Function takes prediction array and picks a message to display to user
def displayPredictionString(prediction):
    firstGuess=np.argmax(prediction)
    firstGuessVal=np.amax(prediction)
    prediction[0,firstGuess]=0
    secondGuess=np.argmax(prediction)
    secondGuessVal=np.amax(prediction)
    prediction[0,secondGuess]=0
    thirdGuess=np.argmax(prediction)
    thirdGuessVal=np.amax(prediction)
    if firstGuessVal+secondGuessVal+thirdGuessVal<0.9:
        resultString.set('I cant tell what that is. Keep in mind I can only read a single digit at a time.')
    elif firstGuessVal>0.9:
        resultString.set('You drew a '+str(firstGuess)+'.')
    elif firstGuessVal+secondGuessVal>0.9:
        resultString.set('I think you drew a '+str(firstGuess)+' but it might be a '+str(secondGuess)+'.')
    else:
        resultString.set('I think you drew a '+str(firstGuess)+' but it might be a '+str(secondGuess)+' or a '+str(thirdGuess)+'.')


#Function feeds converts canvas to array, feeds it into model and displays 
#prediction string
def evaluateCanvas(event=0):
    canvas.postscript(file='canvas.eps',colormode="gray")
    image=imageConverter('canvas.eps')
    Prediction=digitModel.predict(np.reshape(image,(1,28,28,1)))
    numPrediction=np.argmax(Prediction)
    displayPredictionString(Prediction)


#initializing Tkinter widgets
#===========================================================================


#Initializing window
window=tk.Tk()
window.title('Paint')
window.state('normal')
window.configure(bg='#404040')


#Creating toolbar to hold other widgets
toolBar=tk.Frame(window,relief='raised',borderwidth=2)


#Creating canvas on which the user can draw
canvas=tk.Canvas(window,height=600,width=600)
canvas.configure(bg='white')
clear=tk.Button(toolBar, text="Clear All", command=clearScreen)


#Creating thickness slider
thickness=tk.Frame(toolBar)
thicknessLabel=tk.Label(thickness,text='Thickness')
thicknessScale=tk.Scale(thickness,from_=1,to=10,orient='horizontal')


#Creating color choice menu
chooseColor=tk.StringVar()
colors=["Black",
        "White",
        "Red",
        "Blue",
        "Green",
        "Yellow",
        "Purple",
        "Brown"]

chooseColor.set(colors[0])
colorMenu=tk.OptionMenu(toolBar,chooseColor,*colors)


#Creating evaluation button
evaluate=tk.Button(toolBar, text="Evaluate",command=evaluateCanvas)


#Creating results display
resultString=tk.StringVar()
resultString.set("Draw a Number between 0 and 9")
result=tk.Message(toolBar,textvariable=resultString, aspect=250)


#Placing widgets on grid
#===========================================================================


#Placing base widgets onto window
toolBar.grid(row=0,column=0,sticky='nwe')
canvas.grid(row=1,column=0)


#Placing widgets on toolbar
clear.grid(row=0,column=0,sticky='news')
thickness.grid(row=0,column=1,sticky='news')
colorMenu.grid(row=0,column=2,sticky='news')
evaluate.grid(row=0,column=3,sticky='news')
result.grid(row=0,column=4,sticky='news')


#Placing widgets onto thickness frame
thicknessLabel.grid(row=0, column=0)
thicknessScale.grid(row=1,column=0)


#Binding Keys
#===========================================================================


#binding drawing functions to mouse
canvas.bind('<Button-1>',locate_xy)
canvas.bind('<B1-Motion>',draw)


#Binding clear and evaluate commands to keyboard shortcuts
window.bind('<Return>',evaluateCanvas)
window.bind('c',clearScreen)


window.mainloop()
