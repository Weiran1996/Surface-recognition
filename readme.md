#install python environment
conda create -n wx python=3
activate wx
demon\2019-sensors-expo\Slot Car Demo>pip install -r requirements.txt

## wx demos
python -m wx.tools.wxget_docs_demo demo

read the docs
python -m wx.tools.wxget_docs_demo docs


# install wxglade
wxglade is GUI desigh tool for wxPython
load https://github.com/wxGlade/wxGlade/archive/master.zip
unzip to this directory so there is 
Slot Car Demo>python wxGlade-master\wxGlade.py

## wxGlade all widgets demo
Slot Car Demo\wxGlade-master\examples\AllWidgets>python AllWidgets_30_Phoenix.py


# possible useful wx widgets
core windows
                           bitmapbutton
                           button
                           checkbox
                           gauge
                           radiobox
                           slider
                           togglebutton
advanced generic widgets
                           aquabutton
                           knobctrl
                           peakmeter
                           piectrl
                           pygauge
                           shapedbutton
                           speedmeter

# graphics
- https://publicdomainvectors.org/en/free-clipart/
- https://search.creativecommons.org 

## Misc notes
https://towardsdatascience.com/a-very-simple-demo-of-interactive-controls-on-jupyter-notebook-4429cf46aabd
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
