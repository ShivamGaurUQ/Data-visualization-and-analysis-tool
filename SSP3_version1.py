
# coding: utf-8

# <h1><center>Student Feedback Data Analysis</center></h1>

# <img src="analytics.jpg" width="500" height="100" align="center"/>

# In[47]:


#import libraries
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
import xlrd
nltk.download('averaged_perceptron_tagger')

import ipywidgets as widgets
from ipywidgets import HBox, VBox
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
import nltk as nl
nl.download('punkt',quiet=True)
nl.download('stopwords',quiet=True)
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import collections
from nltk.text import Text
import re 
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from operator import itemgetter
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ipywidgets import Button, HBox, VBox, Layout, Box
from IPython.display import display
from IPython.display import clear_output
import pyLDAvis
import pyLDAvis.gensim
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis.sklearn
from pivottablejs import pivot_ui
import os
import sys 
import tkinter as tk
from pandas.api.types import is_string_dtype
import sys
sys.tracebacklimit=0
# Modify the path 
sys.path.append("..")

import yellowbrick as yb
from tkinter import filedialog


# In[108]:


##########################################################################
# Imports
##########################################################################

from yellowbrick.text.base import TextVisualizer

##########################################################################
# PosTagVisualizer
##########################################################################

class PosTagVisualizer(TextVisualizer):
    """
    A part-of-speech tag visualizer colorizes text to enable
    the user to visualize the proportions of nouns, verbs, etc.
    and to use this information to make decisions about text
    normalization (e.g. stemming vs lemmatization) and
    vectorization.

    Parameters
    ----------
    kwargs : dict
        Pass any additional keyword arguments to the super class.
    cmap : dict
        ANSII colormap

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """
    def __init__(self, ax=None, **kwargs):
        
        
        
        super(PosTagVisualizer, self).__init__(ax=ax, **kwargs)

        # TODO: hard-coding in the ANSII colormap for now.
        # Can we let the user reset the colors here?
        self.COLORS = {
            'white'      : "\033[0;37m{}\033[0m",
            'yellow'     : "\033[0;33m{}\033[0m",
            'green'      : "\033[0;32m{}\033[0m",
            'blue'       : "\033[0;34m{}\033[0m",
            'cyan'       : "\033[0;36m{}\033[0m",
            'red'        : "\033[0;31m{}\033[0m",
            'magenta'    : "\033[0;35m{}\033[0m",
            'black'      : "\033[0;30m{}\033[0m",
            'darkwhite'  : "\033[1;37m{}\033[0m",
            'darkyellow' : "\033[1;33m{}\033[0m",
            'darkgreen'  : "\033[1;32m{}\033[0m",
            'darkblue'   : "\033[1;34m{}\033[0m",
            'darkcyan'   : "\033[1;36m{}\033[0m",
            'darkred'    : "\033[1;31m{}\033[0m",
            'darkmagenta': "\033[1;35m{}\033[0m",
            'darkblack'  : "\033[1;30m{}\033[0m",
             None        : "\033[0;0m{}\033[0m"
        }
        
        self.TAGS = {
            'NN'   : 'green',
            'NNS'  : 'green',
            'NNP'  : 'green',
            'NNPS' : 'green',
            'VB'   : 'blue',
            'VBD'  : 'blue',
            'VBG'  : 'blue',
            'VBN'  : 'blue',
            'VBP'  : 'blue',
            'VBZ'  : 'blue',
            'JJ'   : 'red',
            'JJR'  : 'red',
            'JJS'  : 'red',
            'RB'   : 'cyan',
            'RBR'  : 'cyan',
            'RBS'  : 'cyan',
            'IN'   : 'darkwhite',
            'POS'  : 'darkyellow',
            'PRP$' : 'magenta',
            'PRP$' : 'magenta',
            'DT'   : 'black',
            'CC'   : 'black',
            'CD'   : 'black',
            'WDT'  : 'black',
            'WP'   : 'black',
            'WP$'  : 'black',
            'WRB'  : 'black',
            'EX'   : 'yellow',
            'FW'   : 'yellow',
            'LS'   : 'yellow',
            'MD'   : 'yellow',
            'PDT'  : 'yellow',
            'RP'   : 'yellow',
            'SYM'  : 'yellow',
            'TO'   : 'yellow',
            'None' : 'off'
        }
        
        
        
    def colorize(self, token, color):
        
        

        
        return self.COLORS[color].format(token)

    def transform(self, tagged_tuples):
        
        
    
        self.tagged = [
            (self.TAGS.get(tag),tok) for tok, tag in tagged_tuples
        ]


# In[116]:


class SSP:
    def __init__(self):
        self.words = ['Data Settings','Explore','Visualize', 'WordClouds','Topic Modelling','Sentiment Analysis','Text Summarization']
        self.items = [Button(description=w,button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            layout=Layout(width='175px')) for w in self.words]
        display(HBox([item for item in self.items]))
        np.warnings.filterwarnings('ignore')
        self.import_data=widgets.Button(
        description='Import New Data',
        disabled=False,
        button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click to import new data',
        icon='check'
        )
        display(self.import_data)
        self.file_path=""
        self.sheet=""
        self.att=""
        self.org_df=""
        self.import_data.on_click(self.new_data)
        self.items[0].on_click(self.preview_data)
        self.items[1].on_click(self.explore_data)
        self.items[2].on_click(self.vis_data)
        self.items[3].on_click(self.wc_data)
        self.items[4].on_click(self.tm_data)
        self.items[5].on_click(self.sa_data)
        self.items[6].on_click(self.ts_data)
        
        
        
        
        
    
    def new_data(self,b):
        from tkinter import filedialog
        

        root = tk.Tk()
        root.withdraw()

        self.file_path = filedialog.askopenfilename()
        self.labelf=widgets.Label(value="Selected Data : "+str(self.file_path),layout=Layout(width='50%'))
        self.org_df=pd.read_excel(self.file_path)
        clear_output()
        display(HBox([item for item in self.items]))
        display(self.import_data)
        display(self.labelf)
        
        
    
    def preview_data(self,b):
        if len(self.file_path)==0:
            clear_output()
            display(HBox([item for item in self.items]))
            display(self.import_data)
            display("Please import new data")
        else:          
            
            
            clear_output()
            display(HBox([item for item in self.items]))
            display(self.import_data)
            self.wb = xlrd.open_workbook(self.file_path) 
            self.sheets=self.wb.sheet_names()
            self.sheet_to_index={}
            i=0
            while i<len(self.sheets):
                self.sheet_to_index[self.sheets[i]]=i
                i=i+1
            self.sheet_drop=widgets.Dropdown(options=self.sheets,value=self.sheets[0],layout=Layout(width='50%'),description='Sheet:',tooltip='Select a working sheet',disabled=False,)


            self.sheet=self.wb.sheet_by_index(self.sheet_to_index[self.sheet_drop.value])
            self.cols=[]
            for i in range(self.sheet.ncols): 
                self.cols.append(self.sheet.cell_value(0, i)) 
            self.att_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature for data cleaning and analysis',disabled=False,)
            self.att=self.att_drop.value


            self.item_display=[]
            self.item_display.append(self.labelf)
            self.item_display.append(self.sheet_drop)
            self.item_display.append(self.att_drop)
            display(HBox([item for item in self.item_display]))
            
            
            
            

            #print("hi")
            
            
            self.clean_choice=widgets.ToggleButtons(
            options=['Original', 'Cleaned',],    
            value='Cleaned',
            description='Use Data:',
            layout=Layout(width='50%'),
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',

            )
            display(self.clean_choice)
            
            self.df = pd.read_excel(self.file_path)
            self.rem_value=widgets.Text(value='-No Answer-',placeholder='Type missing value indicator',layout=Layout(width='50%'),description='Missing value:',disabled=False)
            if self.clean_choice.value=='Cleaned':

                    self.df_cleaned=self.df.dropna()
                    self.rem_value=widgets.Text(value='-No Answer-',placeholder='Type missing value indicator',layout=Layout(width='50%'),description='Missing value:',disabled=False)
                    display(self.rem_value)
                    self.df_cleaned = self.df_cleaned[self.df_cleaned[self.att_drop.value] != self.rem_value.value]
                    self.df_cleaned = self.df_cleaned.reset_index(drop=True)
                    self.org_df=self.df_cleaned
                    print("\nData cleaned by removing NaN values and rows with value = "+ self.rem_value.value + " in the column " + self.att_drop.value) 
                    print("\nA peek into cleaned data\n")
                    print("\nThere are {} observations and {} features in cleaned dataset \n".format(self.df_cleaned.shape[0],self.df_cleaned.shape[1]))
                    display(self.df_cleaned.head())
                    
                    
                    
                    
            def choice_disp(change):
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display(HBox([item for item in self.item_display]))
                display(self.clean_choice)

                if self.clean_choice.value=='Original':

                    #print("\nThere are {} observations and {} features in original dataset and {} observations and {} features in cleaned dataset. \n".format(df.shape[0],df.shape[1],df_cleaned.shape[0],df_cleaned.shape[1])) 
                    print("\nA peek into originl data\n")
                    print("\nThere are {} observations and {} features in original dataset \n".format(self.df.shape[0],self.df.shape[1]))
                    self.org_df=self.df
                    display(self.df.head())

                else:
                    if self.clean_choice.value=='Cleaned':
                        self.df_cleaned=self.df.dropna()

                        display(self.rem_value)
                        self.df_cleaned = self.df_cleaned[self.df_cleaned[self.att_drop.value] != self.rem_value.value]
                        self.df_cleaned = self.df_cleaned.reset_index(drop=True)
                        self.org_df=self.df_cleaned
                        self.att=self.att_drop.value
                        print("\nData cleaned by removing NaN values and rows with value = "+ self.rem_value.value + " in the column " + self.att_drop.value) 
                        print("\nA peek into cleaned data\n")
                        print("\nThere are {} observations and {} features in cleaned dataset \n".format(self.df_cleaned.shape[0],self.df_cleaned.shape[1]))
                        display(self.df_cleaned.head())



            self.clean_choice.observe(choice_disp,'value')
            self.rem_value.on_submit(choice_disp)
            self.att_drop.observe(choice_disp, 'value')
            
            def sheet_disp(change):
                
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                self.sheet=self.wb.sheet_by_index(self.sheet_to_index[self.sheet_drop.value])
                self.cols=[]
                for i in range(self.sheet.ncols): 
                    self.cols.append(self.sheet.cell_value(0, i)) 
                self.att_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature for data cleaning and analysis',disabled=False,)
                self.att=self.att_drop.value


                self.item_display=[]
                self.item_display.append(self.labelf)
                self.item_display.append(self.sheet_drop)
                self.item_display.append(self.att_drop)
                display(HBox([item for item in self.item_display]))
                self.clean_choice=widgets.ToggleButtons(
                options=['Original', 'Cleaned',],    
                value='Cleaned',
                description='Use Data:',
                layout=Layout(width='50%'),
                disabled=False,
                button_style='success', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Description',

                )
                display(self.clean_choice)

                self.df = pd.read_excel(self.file_path)
                self.rem_value=widgets.Text(value='-No Answer-',placeholder='Type missing value indicator',layout=Layout(width='50%'),description='Missing value:',disabled=False)
                if self.clean_choice.value=='Cleaned':

                        self.df_cleaned=self.df.dropna()
                        self.rem_value=widgets.Text(value='-No Answer-',placeholder='Type missing value indicator',layout=Layout(width='50%'),description='Missing value:',disabled=False)
                        display(self.rem_value)
                        self.df_cleaned = self.df_cleaned[self.df_cleaned[self.att_drop.value] != self.rem_value.value]
                        self.df_cleaned = self.df_cleaned.reset_index(drop=True)
                        self.org_df=self.df_cleaned
                        self.att=self.att_drop.value
                        print("\nData cleaned by removing NaN values and rows with value = "+ self.rem_value.value + " in the column " + self.att_drop.value) 
                        print("\nA peek into cleaned data\n")
                        print("\nThere are {} observations and {} features in cleaned dataset \n".format(self.df_cleaned.shape[0],self.df_cleaned.shape[1]))
                        display(self.df_cleaned.head())
                        
                self.clean_choice.observe(choice_disp,'value')
                self.rem_value.on_submit(choice_disp)
                self.att_drop.observe(choice_disp, 'value')

            
            
            self.sheet_drop.observe(sheet_disp, 'value')
        
    def explore_data(self,b):
                
        if self.sheet==None or self.sheet=="":
            clear_output()
            display(HBox([item for item in self.items]))
            display(self.import_data)
            display("Please change data settings before use")
                
        else: 
            
            clear_output()
            display(HBox([item for item in self.items]))
            display(self.import_data)
            self.action_list=['View Data','Sort Data','Group Data by Feature']
            self.action_drop=widgets.Dropdown(options=self.action_list,value=self.action_list[0],layout=Layout(width='50%'),description='Action:',tooltip='Select an action to perform',disabled=False,)
            display(self.action_drop)
            display(self.org_df)
            
            def action(change):
                flag=0
                def change_order(change):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(self.action_drop)
                    display(self.sort_order)
                    if self.sort_order.value=='Ascending':
                        flag=0
                        self.sort_df = self.org_df.sort_values(by = self.att)
                        display(self.sort_df)
                    else: 
                        flag=1
                        self.sort_df = self.org_df.sort_values(by = self.att,ascending=False)
                        display(self.sort_df)
                        
                        
                
                        
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display(self.action_drop)
                if self.action_drop.value=='View Data':
                    display(self.org_df)
                
                if self.action_drop.value=='Sort Data':
                    self.sort_order=widgets.RadioButtons(
                        options=['Ascending', 'Descending'],
                        value='Ascending',
                        description='Sorting order:',
                        disabled=False
                    )
                    display(self.sort_order)
                    self.sort_df = self.org_df.sort_values(by = self.att)
                    display(self.sort_df)
                    self.sort_order.observe(change_order,'value')
                    
                    
                                            
                if self.action_drop.value=='Group Data by Feature':
                    self.feat = self.org_df.groupby(self.att_drop.value)
                    display(self.feat.describe())
                
                    
                
                
                
                
                
                
            self.action_drop.observe(action, 'value')
            #display(HBox([item for item in self.item_display]))
            
            


    def vis_data(self, b):
        if self.sheet==None or self.sheet=="":
            clear_output()
            display(HBox([item for item in self.items]))
            display(self.import_data)
            display("Please change data settings before use")
                
        else: 
            
            clear_output()
            display(HBox([item for item in self.items]))
            display(self.import_data)
            display(widgets.Label(value="Visualise feature distribution through Bar and Pie Chart",layout=Layout(width='50%')))
            
            
            self.plotfeat_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to plot distribution',disabled=False,)
            display(self.plotfeat_drop)
            entire_feature = self.org_df.groupby(self.plotfeat_drop.value)
            #plot graph of selected distribution
            _labels =self.org_df[self.plotfeat_drop.value].unique()
            plt.figure(figsize=(20,20))
            ax = plt.subplot(221)
            ax.set_aspect(1)

            entire_feature.size().sort_values(ascending=False).plot.pie(labels = _labels, autopct='%1.1f%%',legend = True, fontsize=20)
            plt.ylabel('')
            plt.title(str(self.plotfeat_drop.value)+' Distribution')
            plt.grid(True)
            
            
            plt.figure(figsize=(20,20))
            plt.subplot(222)
            entire_feature.size().sort_values(ascending=False).plot.bar(legend = True,fontsize=20)
            plt.xticks(rotation=50)
            plt.xticks(np.arange(2), _labels)
            plt.ylabel('')
            plt.xlabel('')
            #might need to come up with a better graph title
            plt.title(str(self.plotfeat_drop.value)+' Distribution')
            plt.subplots_adjust(bottom=0.1, right=1.5, top=0.9)
            plt.show()  
            
            def pie_plot(change):
                
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                
                display(widgets.Label(value="Visualise feature distribution through Bar and Pie Chart",layout=Layout(width='50%')))
            
                display(self.plotfeat_drop)
                entire_feature = self.org_df.groupby(self.plotfeat_drop.value)
                #plot graph of selected distribution
                _labels =self.org_df[self.plotfeat_drop.value].unique()
                
                #print(_labels)
                plt.figure(figsize=(20,20))
                ax = plt.subplot(221)
                ax.set_aspect(1)

                entire_feature.size().sort_values(ascending=False).plot.pie(labels = _labels, autopct='%1.1f%%',legend = True, fontsize=20)
                plt.ylabel('')
                plt.title(str(self.plotfeat_drop.value)+' Distribution')
                plt.grid(True)
                
                
              
                plt.figure(figsize=(20,20))
                plt.subplot(222)
                entire_feature.size().sort_values(ascending=False).plot.bar(legend = True,fontsize=20)
                plt.xticks(rotation=50)
                plt.xticks(np.arange(2), _labels)
                plt.ylabel('')
                plt.xlabel('')
                #might need to come up with a better graph title
                plt.title(str(self.plotfeat_drop.value)+' Distribution')
                plt.subplots_adjust(bottom=0.1, right=1.5, top=0.9)
                plt.show()               
                
                
                
            self.plotfeat_drop.observe(pie_plot,'value')
            
            
            
                
    def wc_data(self,b):
        try:
            if self.sheet==None or self.sheet=="":
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display("Please change data settings before use")
            else:
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display(widgets.Label(value="Visualise Text through Word Clouds. Select a feature with text values only.",layout=Layout(width='50%')))


                self.wc_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                display(self.wc_drop)
                #print(self.org_df[self.wc_drop.value].dtype)


                #if is_string_dtype(self.org_df[self.wc_drop.value])==False:
                    #raise ValueError("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")
                all_reviews=self.org_df[self.wc_drop.value]
                #tokenization,removing stopwords, punctuation and stemming
                all_ans=""
                for review in all_reviews:
                    all_ans=all_ans+review+"\n"
                all_ans= all_ans.replace("'", "")
                tokens=word_tokenize(all_ans)
                tokens=[w.lower() for w in tokens]
                text = nl.Text(tokens)
                token_words=[word for word in tokens if word.isalpha()]
                stopword=stopwords.words('english')
                stopword.append('dont')
                stopword.append('didnt')
                stopword.append('doesnt')
                stopword.append('cant')
                stopword.append('couldnt')
                stopword.append('couldve')
                stopword.append('im')
                stopword.append('ive')
                stopword.append('isnt')
                stopword.append('theres')
                stopword.append('wasnt')
                stopword.append('wouldnt')
                stopword.append('a')
                stopword.append('also')
                token_words=[w for w in token_words if not w in stopword]
                porter =PorterStemmer()
                token_stemmed=[porter.stem(w) for w in token_words]
                #clear_output()
                #display(HBox([item for item in items]))
                #creating worlcloud
                #for w in token_words:
                    #if type(w)!="<class 'str'>":
                        #print(type(w))
                        #raise ValueError("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")

                        
                cloudstring=(" ").join(token_words)
            
                wordcloud = WordCloud(max_font_size=50,max_words=100, background_color="black").generate(cloudstring)
                plt.figure(figsize=(20,20))
                ax = plt.subplot(221)
                # plot wordcloud in matplotlib
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title("WordCloud")
                plt.grid(True)
                #plotting bi-gram cloud
                # setup and score the bigrams using the raw frequency.
                finder = BigramCollocationFinder.from_words(token_words)
                bigram_measures = BigramAssocMeasures()
                scored = finder.score_ngrams(bigram_measures.raw_freq)
                scoredList = sorted(scored, key=itemgetter(1), reverse=True)
                word_dict = {} 
                listLen = len(scoredList)
                for i in range(listLen):
                    word_dict['_'.join(scoredList[i][0])] = scoredList[i][1]

                wordCloud = WordCloud(max_font_size=50, max_words=100, background_color="black")
                plt.subplot(222) 
                wordCloud.generate_from_frequencies(word_dict)

                plt.title('Most frequently occurring bigrams connected with an underscore_')
                plt.imshow(wordCloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()

                #plotting frequency distribution
                plt.figure(figsize=(25,5))
                ax = plt.subplot(121)
                freqdist = nl.FreqDist(token_words)
                plt.subplot(121) 
                plt.title("Frequency Distribution of top 50 token words")
                freqdist.plot(50)

                


                def wc_change(change):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(widgets.Label(value="Visualise Text through Word Clouds. Select a feature with text values only.",layout=Layout(width='50%')))


                    #self.wc_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                    display(self.wc_drop)
                    #print(self.org_df[self.wc_drop.value].dtype)

                    #if is_string_dtype(self.org_df[self.wc_drop.value])==False:
                        #raise ValueError("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")
 

                    all_reviews=self.org_df[self.wc_drop.value]
                    #tokenization,removing stopwords, punctuation and stemming
                    all_ans=""
                    for review in all_reviews:
                        all_ans=all_ans+review+"\n"
                    all_ans= all_ans.replace("'", "")
                    tokens=word_tokenize(all_ans)
                    tokens=[w.lower() for w in tokens]
                    text = nl.Text(tokens)
                    token_words=[word for word in tokens if word.isalpha()]
                    stopword=stopwords.words('english')
                    stopword.append('dont')
                    stopword.append('didnt')
                    stopword.append('doesnt')
                    stopword.append('cant')
                    stopword.append('couldnt')
                    stopword.append('couldve')
                    stopword.append('im')
                    stopword.append('ive')
                    stopword.append('isnt')
                    stopword.append('theres')
                    stopword.append('wasnt')
                    stopword.append('wouldnt')
                    stopword.append('a')
                    stopword.append('also')
                    token_words=[w for w in token_words if not w in stopword]
                    porter =PorterStemmer()
                    token_stemmed=[porter.stem(w) for w in token_words]
                    #clear_output()
                    #display(HBox([item for item in items]))
                    #creating worlcloud
                    #for w in token_words:
                        #if type(w)!='str':
                            #raise ValueError("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")

                    cloudstring=(" ").join(token_words)
                    wordcloud = WordCloud(max_font_size=50,max_words=100, background_color="black").generate(cloudstring)
                    plt.figure(figsize=(20,20))
                    ax = plt.subplot(221)
                    # plot wordcloud in matplotlib
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    plt.title("WordCloud")
                    plt.grid(True)
                    #plotting bi-gram cloud
                    # setup and score the bigrams using the raw frequency.
                    finder = BigramCollocationFinder.from_words(token_words)
                    bigram_measures = BigramAssocMeasures()
                    scored = finder.score_ngrams(bigram_measures.raw_freq)
                    scoredList = sorted(scored, key=itemgetter(1), reverse=True)
                    word_dict = {} 
                    listLen = len(scoredList)
                    for i in range(listLen):
                        word_dict['_'.join(scoredList[i][0])] = scoredList[i][1]

                    wordCloud = WordCloud(max_font_size=50, max_words=100, background_color="black")
                    plt.subplot(222) 
                    wordCloud.generate_from_frequencies(word_dict)

                    plt.title('Most frequently occurring bigrams connected with an underscore_')
                    plt.imshow(wordCloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.show()

                    #plotting frequency distribution
                    plt.figure(figsize=(25,5))
                    ax = plt.subplot(121)
                    freqdist = nl.FreqDist(token_words)
                    plt.subplot(121) 
                    plt.title("Frequency Distribution of top 50 token words")
                    freqdist.plot(50)

                    





                self.wc_drop.observe(wc_change,'value')
        except Exception as ve:
            print(ve)
            print("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")
            def wc_change(change):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(widgets.Label(value="Visualise Text through Word Clouds. Select a feature with text values only.",layout=Layout(width='50%')))


                    #self.wc_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                    display(self.wc_drop)
                    #print(self.org_df[self.wc_drop.value].dtype)

                    #if is_string_dtype(self.org_df[self.wc_drop.value])==False:
                        #raise ValueError("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")
 

                    all_reviews=self.org_df[self.wc_drop.value]
                    #tokenization,removing stopwords, punctuation and stemming
                    all_ans=""
                    for review in all_reviews:
                        all_ans=all_ans+review+"\n"
                    all_ans= all_ans.replace("'", "")
                    tokens=word_tokenize(all_ans)
                    tokens=[w.lower() for w in tokens]
                    text = nl.Text(tokens)
                    token_words=[word for word in tokens if word.isalpha()]
                    stopword=stopwords.words('english')
                    stopword.append('dont')
                    stopword.append('didnt')
                    stopword.append('doesnt')
                    stopword.append('cant')
                    stopword.append('couldnt')
                    stopword.append('couldve')
                    stopword.append('im')
                    stopword.append('ive')
                    stopword.append('isnt')
                    stopword.append('theres')
                    stopword.append('wasnt')
                    stopword.append('wouldnt')
                    stopword.append('a')
                    stopword.append('also')
                    token_words=[w for w in token_words if not w in stopword]
                    porter =PorterStemmer()
                    token_stemmed=[porter.stem(w) for w in token_words]
                    #clear_output()
                    #display(HBox([item for item in items]))
                    #creating worlcloud
                    #for w in token_words:
                        #if type(w)!='str':
                            #raise ValueError("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")

                    cloudstring=(" ").join(token_words)
                    wordcloud = WordCloud(max_font_size=50,max_words=100, background_color="black").generate(cloudstring)
                    plt.figure(figsize=(20,20))
                    ax = plt.subplot(221)
                    # plot wordcloud in matplotlib
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    plt.title("WordCloud")
                    plt.grid(True)
                    #plotting bi-gram cloud
                    # setup and score the bigrams using the raw frequency.
                    finder = BigramCollocationFinder.from_words(token_words)
                    bigram_measures = BigramAssocMeasures()
                    scored = finder.score_ngrams(bigram_measures.raw_freq)
                    scoredList = sorted(scored, key=itemgetter(1), reverse=True)
                    word_dict = {} 
                    listLen = len(scoredList)
                    for i in range(listLen):
                        word_dict['_'.join(scoredList[i][0])] = scoredList[i][1]

                    wordCloud = WordCloud(max_font_size=50, max_words=100, background_color="black")
                    plt.subplot(222) 
                    wordCloud.generate_from_frequencies(word_dict)

                    plt.title('Most frequently occurring bigrams connected with an underscore_')
                    plt.imshow(wordCloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.show()

                    #plotting frequency distribution
                    plt.figure(figsize=(25,5))
                    ax = plt.subplot(121)
                    freqdist = nl.FreqDist(token_words)
                    plt.subplot(121) 
                    plt.title("Frequency Distribution of top 50 token words")
                    freqdist.plot(50)

        finally:
            self.wc_drop.observe(wc_change,'value')
            
            

        
            
    def tm_data(self,b):
        try:
            if self.sheet==None or self.sheet=="":
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display("Please change data settings before use")
            else:
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display(widgets.Label(value="Explore topics through topic modelling. Select a feature with text values only.",layout=Layout(width='50%')))
                display(widgets.Label(value="Please be patient while the output is generated. This may take a few moments.",layout=Layout(width='50%')))


                self.tm_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                display(self.tm_drop)
                #print(self.org_df[self.wc_drop.value].dtype)


                #if is_string_dtype(self.org_df[self.wc_drop.value])==False:
                    #raise ValueError("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")
                data=self.org_df[self.tm_drop.value]

                vectorizer = CountVectorizer(min_df=5, max_df=0.9, 
                                     stop_words='english', lowercase=True, 
                                     token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
                data_vectorized = vectorizer.fit_transform(data)
                # Build a Latent Dirichlet Allocation Model
                lda_model = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online')
                lda_Z = lda_model.fit_transform(data_vectorized)

                #clear_output()
                #display(HBox([item for item in items]))
                # Visualize the topics
                pyLDAvis.enable_notebook()
                panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
                display(panel)


                def tm_change(change):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(widgets.Label(value="Explore topics through topic modelling. Select a feature with text values only.",layout=Layout(width='50%')))
                    display(widgets.Label(value="Please be patient while the output is generated. This may take a few moments.",layout=Layout(width='50%')))


                    #self.tm_drop=widgets.Dropdown(options=self.cols,value=self.cols[4],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                    display(self.tm_drop)
                    #print(self.org_df[self.wc_drop.value].dtype)


                    #if is_string_dtype(self.org_df[self.wc_drop.value])==False:
                        #raise ValueError("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")
                    data=self.org_df[self.tm_drop.value]

                    vectorizer = CountVectorizer(min_df=5, max_df=0.9, 
                                         stop_words='english', lowercase=True, 
                                         token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
                    data_vectorized = vectorizer.fit_transform(data)
                    # Build a Latent Dirichlet Allocation Model
                    lda_model = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online')
                    lda_Z = lda_model.fit_transform(data_vectorized)

                    #clear_output()
                    #display(HBox([item for item in items]))
                    # Visualize the topics
                    pyLDAvis.enable_notebook()
                    panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
                    display(panel)

                self.tm_drop.observe(tm_change,'value')
        except Exception as ve:
            print(ve)
            print("That is not a suitable feature for Topic Modelling. Select a feature with text values only.")
            def tm_change(change):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(widgets.Label(value="Explore topics through topic modelling. Select a feature with text values only.",layout=Layout(width='50%')))
                    display(widgets.Label(value="Please be patient while the output is generated. This may take a few moments.",layout=Layout(width='50%')))


                    #self.tm_drop=widgets.Dropdown(options=self.cols,value=self.cols[4],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                    display(self.tm_drop)
                    #print(self.org_df[self.wc_drop.value].dtype)


                    #if is_string_dtype(self.org_df[self.wc_drop.value])==False:
                        #raise ValueError("That is not a suitable feature to generate WordCloud. Select a feature with text values only.")
                    data=self.org_df[self.tm_drop.value]

                    vectorizer = CountVectorizer(min_df=5, max_df=0.9, 
                                         stop_words='english', lowercase=True, 
                                         token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
                    data_vectorized = vectorizer.fit_transform(data)
                    # Build a Latent Dirichlet Allocation Model
                    lda_model = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online')
                    lda_Z = lda_model.fit_transform(data_vectorized)

                    #clear_output()
                    #display(HBox([item for item in items]))
                    # Visualize the topics
                    pyLDAvis.enable_notebook()
                    panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
                    display(panel)

            
        finally:
            self.tm_drop.observe(tm_change,'value')
            

    def sa_data(self,b):
        try:
            if self.sheet==None or self.sheet=="":
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display("Please change data settings before use")
            else:
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display(widgets.Label(value="Opinion mining. Select a feature with text values only.",layout=Layout(width='50%')))
                display(widgets.Label(value="Please be patient while the output is generated. This may take a few moments.",layout=Layout(width='50%')))


                self.sa_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                display(self.sa_drop)
                
                all_reviews=self.org_df[self.sa_drop.value]
                #sentiment analysis
                analyser = SentimentIntensityAnalyzer()
                
                pos_entire=[]
                neg_entire=[]
                neutral_entire=[]

                
                for review in all_reviews:
                    scores = analyser.polarity_scores(review)
                    if scores['compound']<=-0.5:
                        neg_entire.append(review)
                    if scores['compound']>=0.5:
                        pos_entire.append(review)
                    if scores['compound']>-0.5 and scores['compound']<0.5:
                        neutral_entire.append(review)
                #clear_output()
                #display(HBox([item for item in items]))        
                type_length=[len(pos_entire),len(neutral_entire),len(neg_entire)]
                sent_type=['positive','neutral','negative']
                plt.pie(type_length, labels=sent_type, startangle=90, autopct='%.1f%%')
                plt.title('Sentiment distribution')
                plt.show()

                bars1 = [len(pos_entire)]
                bars2 = [len(neutral_entire)]
                bars3 = [len(neg_entire)]
                sent_type=['positive','neutral','negative']

                # set width of bar
                barWidth = 0.25
                # Set position of bar on X axis
                r1 = np.arange(len(bars1))
                r2 = [x + barWidth for x in r1]
                r3 = [x + barWidth for x in r2]

                # Make the plot
                plt.figure(figsize=(15,10))
                plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Positive')
                plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Neutral')
                plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='Negative')

                # Add xticks on the middle of the group bars
                plt.xlabel('Group', fontweight='bold')
                plt.xticks([r + barWidth for r in range(len(bars1))], ['Entire Cohort'])


                # Create legend & Show graphic
                plt.legend()
                plt.title('Sentiment Distributions')
                plt.show()
                
                def sa_change(change):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(widgets.Label(value="Opinion mining. Select a feature with text values only.",layout=Layout(width='50%')))
                    display(widgets.Label(value="Please be patient while the output is generated. This may take a few moments.",layout=Layout(width='50%')))


                    #self.sa_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                    display(self.sa_drop)

                    all_reviews=self.org_df[self.sa_drop.value]
                    #sentiment analysis
                    analyser = SentimentIntensityAnalyzer()

                    pos_entire=[]
                    neg_entire=[]
                    neutral_entire=[]


                    for review in all_reviews:
                        scores = analyser.polarity_scores(review)
                        if scores['compound']<=-0.5:
                            neg_entire.append(review)
                        if scores['compound']>=0.5:
                            pos_entire.append(review)
                        if scores['compound']>-0.5 and scores['compound']<0.5:
                            neutral_entire.append(review)
                    #clear_output()
                    #display(HBox([item for item in items]))        
                    type_length=[len(pos_entire),len(neutral_entire),len(neg_entire)]
                    sent_type=['positive','neutral','negative']
                    plt.pie(type_length, labels=sent_type, startangle=90, autopct='%.1f%%')
                    plt.title('Sentiment distribution')
                    plt.show()

                    bars1 = [len(pos_entire)]
                    bars2 = [len(neutral_entire)]
                    bars3 = [len(neg_entire)]
                    sent_type=['positive','neutral','negative']

                    # set width of bar
                    barWidth = 0.25
                    # Set position of bar on X axis
                    r1 = np.arange(len(bars1))
                    r2 = [x + barWidth for x in r1]
                    r3 = [x + barWidth for x in r2]

                    # Make the plot
                    plt.figure(figsize=(15,10))
                    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Positive')
                    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Neutral')
                    plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='Negative')

                    # Add xticks on the middle of the group bars
                    plt.xlabel('Group', fontweight='bold')
                    plt.xticks([r + barWidth for r in range(len(bars1))], ['Entire Cohort'])


                    # Create legend & Show graphic
                    plt.legend()
                    plt.title('Sentiment Distributions')
                    plt.show()
                    
                
                self.sa_drop.observe(sa_change,'value')
                
        except Exception as ve:
            print(ve)
            print("That is not a suitable feature for Sentiment Analysis. Select a feature with text values only.")
            def sa_change(change):
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display(widgets.Label(value="Opinion mining. Select a feature with text values only.",layout=Layout(width='50%')))
                display(widgets.Label(value="Please be patient while the output is generated. This may take a few moments.",layout=Layout(width='50%')))


                #self.sa_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                display(self.sa_drop)
                
                all_reviews=self.org_df[self.sa_drop.value]
                #sentiment analysis
                analyser = SentimentIntensityAnalyzer()
                
                pos_entire=[]
                neg_entire=[]
                neutral_entire=[]

                
                for review in all_reviews:
                    scores = analyser.polarity_scores(review)
                    if scores['compound']<=-0.5:
                        neg_entire.append(review)
                    if scores['compound']>=0.5:
                        pos_entire.append(review)
                    if scores['compound']>-0.5 and scores['compound']<0.5:
                        neutral_entire.append(review)
                #clear_output()
                #display(HBox([item for item in items]))        
                type_length=[len(pos_entire),len(neutral_entire),len(neg_entire)]
                sent_type=['positive','neutral','negative']
                plt.pie(type_length, labels=sent_type, startangle=90, autopct='%.1f%%')
                plt.title('Sentiment distribution')
                plt.show()

                bars1 = [len(pos_entire)]
                bars2 = [len(neutral_entire)]
                bars3 = [len(neg_entire)]
                sent_type=['positive','neutral','negative']

                # set width of bar
                barWidth = 0.25
                # Set position of bar on X axis
                r1 = np.arange(len(bars1))
                r2 = [x + barWidth for x in r1]
                r3 = [x + barWidth for x in r2]

                # Make the plot
                plt.figure(figsize=(15,10))
                plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Positive')
                plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Neutral')
                plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='Negative')

                # Add xticks on the middle of the group bars
                plt.xlabel('Group', fontweight='bold')
                plt.xticks([r + barWidth for r in range(len(bars1))], ['Entire Cohort'])


                # Create legend & Show graphic
                plt.legend()
                plt.title('Sentiment Distributions')
                plt.show()
                
            
        finally:
            self.sa_drop.observe(sa_change,'value')
            
                
        

    def ts_data(self,b):
        try:
            if self.sheet==None or self.sheet=="":
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display("Please change data settings before use")
            else:
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display(widgets.Label(value="Select a feature with text values only.",layout=Layout(width='50%')))
                display(widgets.Label(value="Please be patient while the output is generated. This may take a few moments.",layout=Layout(width='50%')))


                self.ts_drop=widgets.Dropdown(options=self.cols,value=self.cols[0],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                display(self.ts_drop)
                
                all_reviews=self.org_df[self.ts_drop.value]
                
                
                pos_entire=[]
                neg_entire=[]
                neutral_entire=[]
                analyser = SentimentIntensityAnalyzer()
                for review in all_reviews:
                    scores = analyser.polarity_scores(review)
                    if scores['compound']<=-0.5:
                        neg_entire.append(review)
                    if scores['compound']>=0.5:
                        pos_entire.append(review)
                    if scores['compound']>-0.5 and scores['compound']<0.5:
                        neutral_entire.append(review)
                #sentiment analysis
                #clear_output()
                #display(HBox([item for item in items]))
                self.bpos = widgets.Button(description="Next positive text",button_style='success',layout=Layout(width='175px'))
                self.bneg = widgets.Button(description="Next negative text",button_style='danger',layout=Layout(width='175px'))
                self.bneu = widgets.Button(description="Next neutral text",button_style='info',layout=Layout(width='175px'))
                display(HBox([self.bpos,self.bneu,self.bneg]))
                
                ########################################################################
                
                
                def posref(b):
                    #clear_output()
                    #display(HBox([item for item in items]))
                    #display(HBox([bpos,bneu,bneg]))
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(self.ts_drop)
                    display(HBox([self.bpos,self.bneu,self.bneg]))
                    print("\nPositive Text\n")
                    display(pos_entire[0])
                    text_for_summ=pos_entire[0]
                    article_text = re.sub(r'\[[0-9]*\]', ' ', text_for_summ)  
                    article_text = re.sub(r'\s+', ' ', article_text)  
                    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                    sentence_list = nl.sent_tokenize(article_text)  

                    stopwords = nl.corpus.stopwords.words('english')

                    word_frequencies = {}  
                    for word in nl.word_tokenize(formatted_article_text):  
                        if word not in stopwords:
                            if word not in word_frequencies.keys():
                                word_frequencies[word] = 1
                            else:
                                word_frequencies[word] += 1



                    maximum_frequncy = max(word_frequencies.values())

                    for word in word_frequencies.keys():  
                        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                    sentence_scores = {}  
                    for sent in sentence_list:  
                        for word in nl.word_tokenize(sent.lower()):
                            if word in word_frequencies.keys():
                                if len(sent.split(' ')) < 30:
                                    if sent not in sentence_scores.keys():
                                        sentence_scores[sent] = word_frequencies[word]
                                    else:
                                        sentence_scores[sent] += word_frequencies[word]


                    import heapq  
                    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

                    summary = ' '.join(summary_sentences) 


                    tokens = word_tokenize(summary)
                    tagged = pos_tag(tokens)


                    visualizer = PosTagVisualizer()
                    visualizer.transform(tagged)


                    #print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))




                    display('===================================SUMMARY=========================================================')
                    #display(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                    print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                    print('\n')


                    item_rem=pos_entire.pop(0)
                    pos_entire.append(item_rem)



                self.bpos.on_click(posref)

                def neuref(b):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(self.ts_drop)
                    display(HBox([self.bpos,self.bneu,self.bneg]))
                    print("\nNeutral Text\n")
                    display(neutral_entire[0])
                    text_for_summ=neutral_entire[0]
                    article_text = re.sub(r'\[[0-9]*\]', ' ', text_for_summ)  
                    article_text = re.sub(r'\s+', ' ', article_text)  
                    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                    sentence_list = nl.sent_tokenize(article_text)  

                    stopwords = nl.corpus.stopwords.words('english')

                    word_frequencies = {}  
                    for word in nl.word_tokenize(formatted_article_text):  
                        if word not in stopwords:
                            if word not in word_frequencies.keys():
                                word_frequencies[word] = 1
                            else:
                                word_frequencies[word] += 1



                    maximum_frequncy = max(word_frequencies.values())

                    for word in word_frequencies.keys():  
                        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                    sentence_scores = {}  
                    for sent in sentence_list:  
                        for word in nl.word_tokenize(sent.lower()):
                            if word in word_frequencies.keys():
                                if len(sent.split(' ')) < 30:
                                    if sent not in sentence_scores.keys():
                                        sentence_scores[sent] = word_frequencies[word]
                                    else:
                                        sentence_scores[sent] += word_frequencies[word]


                    import heapq  
                    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

                    summary = ' '.join(summary_sentences) 


                    tokens = word_tokenize(summary)
                    tagged = pos_tag(tokens)


                    visualizer = PosTagVisualizer()
                    visualizer.transform(tagged)










                    display('===================================SUMMARY=========================================================')
                    #display(summary) 
                    print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                    print('\n')












                    item_rem=neutral_entire.pop(0)
                    neutral_entire.append(item_rem)



                self.bneu.on_click(neuref)

                def negref(b):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(self.ts_drop)
                    display(HBox([self.bpos,self.bneu,self.bneg]))
                    print("\nNegative Text\n")
                    display(neg_entire[0])

                    text_for_summ=neg_entire[0]        
                    article_text = re.sub(r'\[[0-9]*\]', ' ', text_for_summ)  
                    article_text = re.sub(r'\s+', ' ', article_text)  
                    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                    sentence_list = nl.sent_tokenize(article_text)  

                    stopwords = nl.corpus.stopwords.words('english')

                    word_frequencies = {}  
                    for word in nl.word_tokenize(formatted_article_text):  
                        if word not in stopwords:
                            if word not in word_frequencies.keys():
                                word_frequencies[word] = 1
                            else:
                                word_frequencies[word] += 1



                    maximum_frequncy = max(word_frequencies.values())

                    for word in word_frequencies.keys():  
                        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                    sentence_scores = {}  
                    for sent in sentence_list:  
                        for word in nl.word_tokenize(sent.lower()):
                            if word in word_frequencies.keys():
                                if len(sent.split(' ')) < 30:
                                    if sent not in sentence_scores.keys():
                                        sentence_scores[sent] = word_frequencies[word]
                                    else:
                                        sentence_scores[sent] += word_frequencies[word]


                    import heapq  
                    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

                    summary = ' '.join(summary_sentences) 


                    tokens = word_tokenize(summary)
                    tagged = pos_tag(tokens)


                    visualizer = PosTagVisualizer()
                    visualizer.transform(tagged)

                    display('===================================SUMMARY=========================================================')
                    #display(summary) 
                    print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                    print('\n')








                    item_rem=neg_entire.pop(0)
                    neg_entire.append(item_rem)



                self.bneg.on_click(negref)
                
                
                def ts_change(change):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(widgets.Label(value="Select a feature with text values only.",layout=Layout(width='50%')))
                    display(widgets.Label(value="Please be patient while the output is generated. This may take a few moments.",layout=Layout(width='50%')))


                    #self.ts_drop=widgets.Dropdown(options=self.cols,value=self.cols[4],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                    display(self.ts_drop)

                    all_reviews=self.org_df[self.ts_drop.value]


                    pos_entire=[]
                    neg_entire=[]
                    neutral_entire=[]
                    analyser = SentimentIntensityAnalyzer()
                    for review in all_reviews:
                        scores = analyser.polarity_scores(review)
                        if scores['compound']<=-0.5:
                            neg_entire.append(review)
                        if scores['compound']>=0.5:
                            pos_entire.append(review)
                        if scores['compound']>-0.5 and scores['compound']<0.5:
                            neutral_entire.append(review)
                    #sentiment analysis
                    #clear_output()
                    #display(HBox([item for item in items]))
                    self.bpos = widgets.Button(description="Next positive text",button_style='success',layout=Layout(width='175px'))
                    self.bneg = widgets.Button(description="Next negative text",button_style='danger',layout=Layout(width='175px'))
                    self.bneu = widgets.Button(description="Next neutral text",button_style='info',layout=Layout(width='175px'))
                    display(HBox([self.bpos,self.bneu,self.bneg]))

                    ########################################################################


                    def posref(b):
                        #clear_output()
                        #display(HBox([item for item in items]))
                        #display(HBox([bpos,bneu,bneg]))
                        clear_output()
                        display(HBox([item for item in self.items]))
                        display(self.import_data)
                        display(self.ts_drop)
                        display(HBox([self.bpos,self.bneu,self.bneg]))
                        print("\nPositive Text\n")
                        display(pos_entire[0])
                        text_for_summ=pos_entire[0]
                        article_text = re.sub(r'\[[0-9]*\]', ' ', text_for_summ)  
                        article_text = re.sub(r'\s+', ' ', article_text)  
                        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                        sentence_list = nl.sent_tokenize(article_text)  

                        stopwords = nl.corpus.stopwords.words('english')

                        word_frequencies = {}  
                        for word in nl.word_tokenize(formatted_article_text):  
                            if word not in stopwords:
                                if word not in word_frequencies.keys():
                                    word_frequencies[word] = 1
                                else:
                                    word_frequencies[word] += 1



                        maximum_frequncy = max(word_frequencies.values())

                        for word in word_frequencies.keys():  
                            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                        sentence_scores = {}  
                        for sent in sentence_list:  
                            for word in nl.word_tokenize(sent.lower()):
                                if word in word_frequencies.keys():
                                    if len(sent.split(' ')) < 30:
                                        if sent not in sentence_scores.keys():
                                            sentence_scores[sent] = word_frequencies[word]
                                        else:
                                            sentence_scores[sent] += word_frequencies[word]


                        import heapq  
                        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

                        summary = ' '.join(summary_sentences) 


                        tokens = word_tokenize(summary)
                        tagged = pos_tag(tokens)


                        visualizer = PosTagVisualizer()
                        visualizer.transform(tagged)


                        #print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))




                        display('===================================SUMMARY=========================================================')
                        #display(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                        print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                        print('\n')


                        item_rem=pos_entire.pop(0)
                        pos_entire.append(item_rem)



                    self.bpos.on_click(posref)

                    def neuref(b):
                        clear_output()
                        display(HBox([item for item in self.items]))
                        display(self.import_data)
                        display(self.ts_drop)
                        display(HBox([self.bpos,self.bneu,self.bneg]))
                        print("\nNeutral Text\n")
                        display(neutral_entire[0])
                        text_for_summ=neutral_entire[0]
                        article_text = re.sub(r'\[[0-9]*\]', ' ', text_for_summ)  
                        article_text = re.sub(r'\s+', ' ', article_text)  
                        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                        sentence_list = nl.sent_tokenize(article_text)  

                        stopwords = nl.corpus.stopwords.words('english')

                        word_frequencies = {}  
                        for word in nl.word_tokenize(formatted_article_text):  
                            if word not in stopwords:
                                if word not in word_frequencies.keys():
                                    word_frequencies[word] = 1
                                else:
                                    word_frequencies[word] += 1



                        maximum_frequncy = max(word_frequencies.values())

                        for word in word_frequencies.keys():  
                            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                        sentence_scores = {}  
                        for sent in sentence_list:  
                            for word in nl.word_tokenize(sent.lower()):
                                if word in word_frequencies.keys():
                                    if len(sent.split(' ')) < 30:
                                        if sent not in sentence_scores.keys():
                                            sentence_scores[sent] = word_frequencies[word]
                                        else:
                                            sentence_scores[sent] += word_frequencies[word]


                        import heapq  
                        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

                        summary = ' '.join(summary_sentences) 


                        tokens = word_tokenize(summary)
                        tagged = pos_tag(tokens)


                        visualizer = PosTagVisualizer()
                        visualizer.transform(tagged)










                        display('===================================SUMMARY=========================================================')
                        #display(summary) 
                        print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                        print('\n')












                        item_rem=neutral_entire.pop(0)
                        neutral_entire.append(item_rem)



                    self.bneu.on_click(neuref)

                    def negref(b):
                        clear_output()
                        display(HBox([item for item in self.items]))
                        display(self.import_data)
                        display(self.ts_drop)
                        display(HBox([self.bpos,self.bneu,self.bneg]))
                        print("\nNegative Text\n")
                        display(neg_entire[0])

                        text_for_summ=neg_entire[0]        
                        article_text = re.sub(r'\[[0-9]*\]', ' ', text_for_summ)  
                        article_text = re.sub(r'\s+', ' ', article_text)  
                        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                        sentence_list = nl.sent_tokenize(article_text)  

                        stopwords = nl.corpus.stopwords.words('english')

                        word_frequencies = {}  
                        for word in nl.word_tokenize(formatted_article_text):  
                            if word not in stopwords:
                                if word not in word_frequencies.keys():
                                    word_frequencies[word] = 1
                                else:
                                    word_frequencies[word] += 1



                        maximum_frequncy = max(word_frequencies.values())

                        for word in word_frequencies.keys():  
                            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                        sentence_scores = {}  
                        for sent in sentence_list:  
                            for word in nl.word_tokenize(sent.lower()):
                                if word in word_frequencies.keys():
                                    if len(sent.split(' ')) < 30:
                                        if sent not in sentence_scores.keys():
                                            sentence_scores[sent] = word_frequencies[word]
                                        else:
                                            sentence_scores[sent] += word_frequencies[word]


                        import heapq  
                        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

                        summary = ' '.join(summary_sentences) 


                        tokens = word_tokenize(summary)
                        tagged = pos_tag(tokens)


                        visualizer = PosTagVisualizer()
                        visualizer.transform(tagged)

                        display('===================================SUMMARY=========================================================')
                        #display(summary) 
                        print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                        print('\n')








                        item_rem=neg_entire.pop(0)
                        neg_entire.append(item_rem)



                    self.bneg.on_click(negref)
                    self.ts_drop.observe(ts_change,'value')
                
        except Exception as ve:
            print(ve)
            print("That is not a suitable feature for Sentiment Analysis. Select a feature with text values only.")
            def ts_change(change):
                clear_output()
                display(HBox([item for item in self.items]))
                display(self.import_data)
                display(widgets.Label(value="Select a feature with text values only.",layout=Layout(width='50%')))
                display(widgets.Label(value="Please be patient while the output is generated. This may take a few moments.",layout=Layout(width='50%')))


                #self.ts_drop=widgets.Dropdown(options=self.cols,value=self.cols[4],layout=Layout(width='50%'),description='Feature:',tooltip='Select a feature to generate wordcloud',disabled=False,)
                display(self.ts_drop)
                
                all_reviews=self.org_df[self.ts_drop.value]
                
                
                pos_entire=[]
                neg_entire=[]
                neutral_entire=[]
                analyser = SentimentIntensityAnalyzer()
                for review in all_reviews:
                    scores = analyser.polarity_scores(review)
                    if scores['compound']<=-0.5:
                        neg_entire.append(review)
                    if scores['compound']>=0.5:
                        pos_entire.append(review)
                    if scores['compound']>-0.5 and scores['compound']<0.5:
                        neutral_entire.append(review)
                #sentiment analysis
                #clear_output()
                #display(HBox([item for item in items]))
                self.bpos = widgets.Button(description="Next positive text",button_style='success',layout=Layout(width='175px'))
                self.bneg = widgets.Button(description="Next negative text",button_style='danger',layout=Layout(width='175px'))
                self.bneu = widgets.Button(description="Next neutral text",button_style='info',layout=Layout(width='175px'))
                display(HBox([self.bpos,self.bneu,self.bneg]))
                
                ########################################################################
                
                
                def posref(b):
                    #clear_output()
                    #display(HBox([item for item in items]))
                    #display(HBox([bpos,bneu,bneg]))
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(self.ts_drop)
                    display(HBox([self.bpos,self.bneu,self.bneg]))
                    print("\nPositive Text\n")
                    display(pos_entire[0])
                    text_for_summ=pos_entire[0]
                    article_text = re.sub(r'\[[0-9]*\]', ' ', text_for_summ)  
                    article_text = re.sub(r'\s+', ' ', article_text)  
                    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                    sentence_list = nl.sent_tokenize(article_text)  

                    stopwords = nl.corpus.stopwords.words('english')

                    word_frequencies = {}  
                    for word in nl.word_tokenize(formatted_article_text):  
                        if word not in stopwords:
                            if word not in word_frequencies.keys():
                                word_frequencies[word] = 1
                            else:
                                word_frequencies[word] += 1



                    maximum_frequncy = max(word_frequencies.values())

                    for word in word_frequencies.keys():  
                        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                    sentence_scores = {}  
                    for sent in sentence_list:  
                        for word in nl.word_tokenize(sent.lower()):
                            if word in word_frequencies.keys():
                                if len(sent.split(' ')) < 30:
                                    if sent not in sentence_scores.keys():
                                        sentence_scores[sent] = word_frequencies[word]
                                    else:
                                        sentence_scores[sent] += word_frequencies[word]


                    import heapq  
                    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

                    summary = ' '.join(summary_sentences) 


                    tokens = word_tokenize(summary)
                    tagged = pos_tag(tokens)


                    visualizer = PosTagVisualizer()
                    visualizer.transform(tagged)


                    #print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))




                    display('===================================SUMMARY=========================================================')
                    #display(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                    print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                    print('\n')


                    item_rem=pos_entire.pop(0)
                    pos_entire.append(item_rem)



                self.bpos.on_click(posref)

                def neuref(b):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(self.ts_drop)
                    display(HBox([self.bpos,self.bneu,self.bneg]))
                    print("\nNeutral Text\n")
                    display(neutral_entire[0])
                    text_for_summ=neutral_entire[0]
                    article_text = re.sub(r'\[[0-9]*\]', ' ', text_for_summ)  
                    article_text = re.sub(r'\s+', ' ', article_text)  
                    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                    sentence_list = nl.sent_tokenize(article_text)  

                    stopwords = nl.corpus.stopwords.words('english')

                    word_frequencies = {}  
                    for word in nl.word_tokenize(formatted_article_text):  
                        if word not in stopwords:
                            if word not in word_frequencies.keys():
                                word_frequencies[word] = 1
                            else:
                                word_frequencies[word] += 1



                    maximum_frequncy = max(word_frequencies.values())

                    for word in word_frequencies.keys():  
                        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                    sentence_scores = {}  
                    for sent in sentence_list:  
                        for word in nl.word_tokenize(sent.lower()):
                            if word in word_frequencies.keys():
                                if len(sent.split(' ')) < 30:
                                    if sent not in sentence_scores.keys():
                                        sentence_scores[sent] = word_frequencies[word]
                                    else:
                                        sentence_scores[sent] += word_frequencies[word]


                    import heapq  
                    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

                    summary = ' '.join(summary_sentences) 


                    tokens = word_tokenize(summary)
                    tagged = pos_tag(tokens)


                    visualizer = PosTagVisualizer()
                    visualizer.transform(tagged)










                    display('===================================SUMMARY=========================================================')
                    #display(summary) 
                    print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                    print('\n')












                    item_rem=neutral_entire.pop(0)
                    neutral_entire.append(item_rem)



                self.bneu.on_click(neuref)

                def negref(b):
                    clear_output()
                    display(HBox([item for item in self.items]))
                    display(self.import_data)
                    display(self.ts_drop)
                    display(HBox([self.bpos,self.bneu,self.bneg]))
                    print("\nNegative Text\n")
                    display(neg_entire[0])

                    text_for_summ=neg_entire[0]        
                    article_text = re.sub(r'\[[0-9]*\]', ' ', text_for_summ)  
                    article_text = re.sub(r'\s+', ' ', article_text)  
                    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
                    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
                    sentence_list = nl.sent_tokenize(article_text)  

                    stopwords = nl.corpus.stopwords.words('english')

                    word_frequencies = {}  
                    for word in nl.word_tokenize(formatted_article_text):  
                        if word not in stopwords:
                            if word not in word_frequencies.keys():
                                word_frequencies[word] = 1
                            else:
                                word_frequencies[word] += 1



                    maximum_frequncy = max(word_frequencies.values())

                    for word in word_frequencies.keys():  
                        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

                    sentence_scores = {}  
                    for sent in sentence_list:  
                        for word in nl.word_tokenize(sent.lower()):
                            if word in word_frequencies.keys():
                                if len(sent.split(' ')) < 30:
                                    if sent not in sentence_scores.keys():
                                        sentence_scores[sent] = word_frequencies[word]
                                    else:
                                        sentence_scores[sent] += word_frequencies[word]


                    import heapq  
                    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

                    summary = ' '.join(summary_sentences) 


                    tokens = word_tokenize(summary)
                    tagged = pos_tag(tokens)


                    visualizer = PosTagVisualizer()
                    visualizer.transform(tagged)

                    display('===================================SUMMARY=========================================================')
                    #display(summary) 
                    print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
                    print('\n')








                    item_rem=neg_entire.pop(0)
                    neg_entire.append(item_rem)



                self.bneg.on_click(negref)
                
        finally:
            self.ts_drop.observe(ts_change,'value')
    

    


# In[117]:


def main():
    obj=SSP()
   


# In[118]:


if __name__=='__main__':
    np.warnings.filterwarnings('ignore')
    main()

