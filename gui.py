import PySimpleGUI as sg
from tools import predict_from_list
layout = [  [sg.Text('Input your pass phrase:'), sg.InputText('correct horse battery staple',key='_INPUT_')],
            [sg.OK(), sg.Exit()]]

# Create the Window
window = sg.Window('Rhyming Passphrase Generator', layout)
# Event Loop to process "events"
while True:             
    event, values = window.Read()
    if event in (None, 'Exit'):
        break
    if event in ('OK'):
        strings = predict_from_list('./model.h5',values["_INPUT_"].split(' '))
        sg.popup( "Your resulting rhyming password is: " + ' '.join(strings), title= "Results")
window.Close()