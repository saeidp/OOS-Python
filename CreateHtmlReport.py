import nbconvert

oosVerification = !jupyter nbconvert --to html "OOSVerification.ipynb" --stdout

from IPython.display import HTML, display
display(HTML('\n'.join(oosVerification)))