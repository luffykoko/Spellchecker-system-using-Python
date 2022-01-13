        #NLTK
                                    #Import Lirbraries#
#Read PDF
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from io import StringIO
from pdfminer.pdfpage import PDFPage

# IMport operator library
import operator

#NLTK_Token_and_sent
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

#Edit_distance
import string
from collections import Counter
import numpy as np

#Pre-Processing
import contractions
import re
from bs4 import BeautifulSoup
import unicodedata

#NLTK_N_Gram
from nltk import ngrams
import collections

#TextBlob_N_Gram
from textblob import TextBlob

#GUI libraries
from tkinter import *
import tkinter.scrolledtext as scroll

root = Tk()
#N-Gram
def n_gram_NLTK(blob):
    n = 2
    unigrams = ngrams(blob.split(), n)
    bairamFreq = collections.Counter(unigrams)

    print("\n\n",bairamFreq.most_common(700))

    return bairamFreq

def N_Gram_Comparission(listToStr,dictionary, abbreviations ):


    #NLTK:
    User_N_Gram = n_gram_NLTK(listToStr)
    dictionary = ' '.join([str(elem) for elem in dictionary])
    Dictiinary_N_Gram = n_gram_NLTK(dictionary)

    new_final_output=[]

    REALWORD = []
    for ngram in User_N_Gram:
        ngram = ' '.join([str(elem) for elem in ngram])
        #print("\n\n", ngram)
        if ngram in dictionary:
            new_final_output.append(ngram)

        elif ngram in abbreviations: 
            new_final_output.append(ngram)
        else:

            REALWORD.append(ngram) 
    
    return new_final_output, REALWORD

global dictionary


global wrong_token
wrong_token = []

#Text Pre-Processing#
def remove_accents(input_str):

  only_ascii = unicodedata.normalize('NFKD', input_str).encode('ASCII', 'ignore').decode('utf-8', 'ignore')

  return only_ascii


def text_pre_processing (text):

    print("\n\n************************************")
    print("*** Text Pre-Processing and Cleaning ***")
    print("************************************\n\n")

        #1) Contraction to Expansion
    print("\n\n\t\t *** 1- Convert the shortened to orignal form *** \n\n")
    expanded_words = []    
    for word in text.split():
        expanded_words.append(contractions.fix(word)) # using contractions.fix to expand the shortened words
    expanded_text = ' '.join(expanded_words)
    #print('Original text: ' + text)
    #print('Expanded_text: ' + expanded_text)

        #2) Count and Remove Emails from text
    print("\n\n\t\t *** 2- Count and Remove Emails from text *** \n\n")
    #2.1) Count Emails from text 
    Emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', expanded_text)
    print("\n\n All the Emails found in the text: ",Emails)
    #2.2) Remove Emails from text 
    line = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', '', expanded_text)
    #2.3) Print all the emails in the text after deleting it to make sure there is no email in the text anymore
    Emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', line)
    print("\n\n All the emails in the text after deleting it to make sure there is no email in the text anymore: ",Emails)

        #3) Count and Remove URLs from text File
    print("\n\n\t\t *** 3- Count and Remove URLs from text *** \n\n")
    #3.1) Count URLs from text   
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
    print("\n\n All the Emails found in the text: ",urls)
    #3.2) Remove URLs from text 
    line_2 = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', line)
    #3.3) Print all the URLs in the text after deleting it to make sure there is no email in the text anymore
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line_2)
    print("\n\n All the URLs in the text after deleting it to make sure there is no email in the text anymore: ",urls)

        #4) #Remove HTML Tags
    print("\n\n\t\t *** 4- Remove HTML Tags from text *** \n\n")
    text = BeautifulSoup(line_2, 'lxml').get_text()

        #5) Convert Accent Chars
    print("\n\n\t\t *** 5- Convert Accent Chars *** \n\n")
    text = remove_accents(text)

        #6) Replace the ` to '
    print("\n\n\t\t *** 6- Replace the ` to ' *** \n\n")
    text = text.replace("`","'")# firstly we need to repalce the '`' with '''

        #7) Count and Remove all the numbers
    print("\n\n\t\t *** 7- Remove all the numbers and add numbers 1 - 100000 *** \n\n")
    #Remove all the numbers 
    numbers = re.findall("[0-9]+", text)
    print ("\n\n\t\t all the numbers: \n", numbers, "\n")
    text = re.sub("[0-9]+", '',text)

        #8) Remove all punctuation and add new lise 
    print("\n\n\t\t *** 8- Remove all punctuation *** \n\n")
    text = re.sub(r'[^\w\s]','',text)
    #Add new punctuation
    '''
    punctuation = '!"#$%&\'()*+-/:,;.<=>?@[\]^_`{|}~'
    text = text + punctuation
    '''

        #9) Remove Extra Spaces
    print("\n\n\t\t *** 9- Remove Extra Spaces, to make the text ready to split to sentence. *** \n\n")
    text = re.sub("\s\s+", " ", text)


        #10) create new dic for find all the abbreviations
    print("\n\n\t\t *** 11- Remove all punctuation to make the text ready to find all the abbreviations. *** \n\n")
    text1 = text

        #11) Create list have all the abbreviations 
    print("\n\n\t\t *** 12- Create list have all the abbreviations. *** \n\n")
    abbreviations = ' '.join(w for w in (word_tokenize(text1)) if w.isupper() and len(w)>1)
    abbreviations = word_tokenize(abbreviations)
    abbreviations = set(abbreviations) #to make sure there are no any duplicat in the abbreviations 
    print ("\n\n\t\t all the abbreviations: \n", abbreviations, "\n")
    
    text = ' '.join(w for w in (word_tokenize(text)) if not (w.isupper() and len(w)>1)) # to remove all the abbreviations from our dictionary
    global low_dic

        #12) Create new dictionary have all the text in lower case
    print("\n\n\t\t *** 13- Create new dictionary have all the text in lower case. *** \n\n")
    low_dic = [docs.lower() for docs in (word_tokenize(text))] 
    #print ("\n\n\t\t lower dictinary \n\n", low_dic)

    low_dic = ' '.join([str(elem) for elem in low_dic])
    text = low_dic #to add the new lower case dictinary to text dictinary, to enable detection of uppercase non-word erorr
    

        #13) Create new list stemming and lemmatizing
    '''
    No need to do it bc we are doing spelling correction.
    '''
    print("\n\n************************************")
    print("*** Text Pre-Processing and Cleaning Done Successfully ***")
    print("************************************\n\n")
    
    return text,abbreviations
    #global dic2
    #dic2=low_dic
                                    #Read PDF File#

def get_pdf_file_content(path_to_pdf):
    print("\n\n*************************")
    print("*** Reading the PDF File *** ")
    print("*************************\n\n")
                            #A) Read PDF File and Extract the Text
    resource_manager = PDFResourceManager(caching=True)
    out_text = StringIO()
    laParams = LAParams()

    text_converter = TextConverter(resource_manager, out_text, laparams=laParams)
    fp = open(path_to_pdf, 'rb')
    interpreter = PDFPageInterpreter(resource_manager, text_converter)

    for page in PDFPage.get_pages(fp, pagenos=set(), maxpages=0, password="", caching=True, check_extractable=True):
        interpreter.process_page(page)
    
    #Final Variable that Have all the text in PDF File 
    text = out_text.getvalue()

    fp.close()
    text_converter.close()
    out_text.close()



    print("\n\n***************************************************************************")
    print("*** The sentences 200 to 210 from PDF file after convert from PDF to text is: ***")
    print("***************************************************************************\n\n")

    global Set_nltk
    text_nltk_SENT = sent_tokenize(text)
    counter = 1
    for sent in text_nltk_SENT[200:210]:
        print("Sentence number.", counter, "\n\n", sent)
        counter = counter + 1 
    text_nltk_TOKN = word_tokenize(text)
    Set_nltk_TOKN = set(text_nltk_TOKN)

    print("\n\n*************************************************************************")
    print("*** General Information about PDF file after convert from PDF to text is: ***")
    print("*************************************************************************\n\n")
    
    print(f"There are {len(text_nltk_TOKN)} total Tokens in the corpus using NLTK") 
    print(f"There are {len(Set_nltk_TOKN)} unique Tokens in the vocabulary using NLTK")

                            #B) Pre-Processing and Cleaning the text
    text_after_pre_processing, abbreviations  = text_pre_processing(text)

                            #C) Tokenization 
    print("\n\n*************************")
    print("*** Tokenization the Text using NLTK with General Information about Text *** ")
    print("*************************\n\n")
    #USing NLTK to tokenize the text

    text_nltk = word_tokenize(text_after_pre_processing)
    #print("\n\t\t The Doc text file :\n", text_nltk)
    Set_nltk = set(text_nltk)
    print(f"There are {len(text_nltk)} total words in the corpus for NLTK using") 
    print(f"There are {len(Set_nltk)} unique words in the vocabulary for NLTK using")

    print("\n\t\t The Token number 5000 to 5050 tokens for text_nltk file :\n")
    for token in text_nltk[5000:5050]:#for loop to print all the token in the doc file, to see how doc count token in up
        print(token)

    return text_nltk, abbreviations

#User Input#
def user_text_pre_processing (text):

        #1) Contraction to Expansion
    expanded_words = []    
    for word in text.split():
        expanded_words.append(contractions.fix(word)) # using contractions.fix to expand the shotened words
    text = ' '.join(expanded_words)

        #2) Count and Keep the Emails from text
    #2.1) Count Emails from text 
    Emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    print("\n\nUser Input Emails: ",Emails)

        #3) Count and Keep the Emails from text
    #3.1) Count URLs from text   
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    print("\n\nUser Input Urls: ",urls)


    EmailsandUrls = Emails + urls
    EmailsandUrls = ' '.join([str(elem) for elem in EmailsandUrls])
    print("\n\nAll the User Input Emails and Urls: ", EmailsandUrls)

        #4) #Remove HTML Tags
    text = BeautifulSoup(text, 'lxml').get_text()

        #5) Convert Accent Chars
    text = remove_accents(text)

        #6) Replace the ` to '
    text = text.replace("`","'")# firstly we need to repalce the '`' with '''

        #7) Count all punctuation without .,
    punctuation = re.findall(r'[^\w\s]', text)
    print ("\n\n\t\t all the punctuation: \n", punctuation, "\n")
    text = re.sub(r'[^\w\s]', '',text)
   


        #8) Convert all the user text to lower case
    low_dic = [docs.lower() for docs in (word_tokenize(text))] 
    #print ("\n\n\t\t lower dictinary \n\n", low_dic)
    low_dic = ' '.join([str(elem) for elem in low_dic])
    text = low_dic 

    return text, EmailsandUrls

def User_Input_Sent():
                            #A) user input
    user_input2.configure(state='normal')
    user_input2.delete('1.0', END)
    #user_input = input("\n\npleace input your text here: ")
    user_input = user_input1.get('1.0', 'end-1c')#Accept input from GUI text box
                           
                            #B) Pre-Processing and Cleaning the customer text

    user_input2.insert(INSERT, user_input)
    user_input2.configure(state='disabled')

    user_input, EmailsandUrls = user_text_pre_processing (user_input)

                            #C) Tokenization 
    user_input = word_tokenize(user_input)

    return user_input, EmailsandUrls


def non_words_spellCorrection(token):
    non_words_spellCorrect = []
    delete = [left  + right[1:] for left,right in ([(token[:i], token[i:]) for i in range(len(token) + 1)]) if right]
    print("\n\n\t Delete: ",delete)
    swap = [left + right[1] + right[0] + right[2:] for left, right in ([(token[:i], token[i:]) for i in range(len(token) + 1)]) if len(right)>1]
    print("\n\n\t swap: ",swap)
    letters = string.ascii_lowercase
    replace = [left + c + right[1:] for left, right in ([(token[:i], token[i:]) for i in range(len(token) + 1)]) if right for c in letters]
    print("\n\n\t replace: ",replace)    
    insert = [left + c + right for left, right in ([(token[:i], token[i:]) for i in range(len(token) + 1)]) for c in letters]
    print("\n\n\t insert: ",insert)        
    non_words_spellCorrect.extend(delete + swap + replace + insert)
    return non_words_spellCorrect

def Correction_Tokens(token):
    global best_guesses
    best_guesses = []  
    word_counts = Counter(dictionary)
    total_word_count = float(sum(word_counts.values()))
    word_probas = {word: word_counts[word] / total_word_count for word in word_counts.keys()}# to create word probability
    suggestions = set(non_words_spellCorrection(token)) or [token]
    print("\n\n\t\t***suggestions****\n\n",suggestions, "\n\n")
    best_guesses = [w for w in suggestions if w in dictionary]


    print("\n\n\t\t***best_guesses****\n\n",best_guesses, "\n\n")

    return [(w , word_probas[w]) for w in best_guesses]


def CorrectSpelling(token,dictionary):

    if token in dictionary:
        print(f" {token} is already correctly spelt")
        return token
    elif token.isdigit():
        print(f" {token} is numbers")
        return token
    else:
        if token not in dictionary:

            wrong_token.append(token)
            print("\n\n\n  wrong_token wrong_token wrong_token ", wrong_token)

            #this is to highlight the wrong words red
            user_input1.tag_config("red_tag", foreground="red")
            user_input1.tag_remove("red_tag", "1.0", END)


            countvar = IntVar()
            for word in wrong_token:
                pos = "1.0"
                pattern = r"\m{}\M".format(word)
                while user_input1.compare(pos, "<", "end"):
                    pos = user_input1.search(pattern, pos, "end", count=countvar, regexp=True)
                    if pos:
                        pos_end = user_input1.index("{} + {} chars".format(pos, countvar.get()))
                        user_input1.tag_add("red_tag", pos, pos_end)
                        pos = pos_end
                    else:
                        break

        corrections = Correction_Tokens(token)
        print("\n\n\t\t***corrections****\n\n", corrections, "\n\n")
        if corrections:
            probs = np.array([c[1] for c in corrections])
            best_ix = np.argmax(probs)
            correct = corrections[best_ix][0]
            print("\n\n\t\t***correct 1 ****\n\n", correct, "\n\n")
            print("\n\n", f"{correct} is the correct for {token}")

            return correct


#if the selected word is an error, collect index range of the word.
def text_selected():
    if wrong_token:
        global selection_ind
        selection_ind = user_input1.tag_ranges(SEL)
        if selection_ind:
            return True
        else:
            return False
    else:
        return False


#IMplementation of the Levenshtein distance
def edit_levens(text,dictionary2):

    #Implementation of Levenshtein Edit Distance with dynamic programming
    #Inspired by my classmate Lucas Ting Leck King,
    # who helped alot in our programming issues
    #despite note being in our group.
    #This was modified from one of our lab tutorials for this application.
    candidates = {}
    for word in dictionary2:
        n = len(word)
        m = len(text)
        dp = np.zeros((m + 1, n + 1),
                      dtype='int')  
        dp[0, :] = np.arange(0, n + 1)  
        dp[:, 0] = np.arange(0, m + 1)  

    #Dynamic programming implementation 

        for i in range(1, dp.shape[0]):  #column
            for j in range(1, dp.shape[1]): #row
                dp[i, j] = min(
                    dp[i - 1, j] + 1, #insertion
                    dp[i, j - 1] + 1, #deletion
                    dp[i - 1, j - 1] + 0 if text[i - 1] == word[j - 1] else dp[i - 1, j - 1] + 2) #substitution

        candidates[word] = dp[-1][-1]
    candidates = dict(sorted(candidates.items(), key=operator.itemgetter(1))[:5])

    return candidates

#the popup function to be called when right click on selected words
def popup(event):

    if text_selected():
        selected = user_input1.get(*selection_ind)
        candidates = edit_levens(selected, dictionary2)
        candidates_list = list(candidates.items())
        if selected in wrong_token:
            # Clearing the popup menu
            popup_menu.delete(0, END)
            for i in candidates_list:
                #https://stackoverflow.com/questions/54991337/adding-menu-items-to-a-tkinter-menu-in-a-loop-results-in-the-last-menu-item-bein
                #@
                popup_menu.add_command(label=i, command=lambda boundi=i: choose_correction(str(boundi), selection_ind))
                popup_menu.add_separator()
            popup_menu.add_command(label="Add to dictionary",
                                   command=lambda: add_to_dict(selected))
            popup_menu.tk_popup(event.x_root, event.y_root)
            popup_menu.grab_release()

#this function to replace wrong word with correct word
def choose_correction(correction, selection_ind):
    user_input1.delete( *selection_ind)
    correction = re.sub (r'[0-9()\', ]', "", correction)
    user_input1.insert(INSERT, correction)

#Main Function
def main_Function():

    while True:
        user_input, EmailsandUrls = User_Input_Sent()
        lengh_sent = len(user_input)


        new_token_list = []

        token_number = 0
        for token in user_input:

            new_token = CorrectSpelling(token,dictionary)
            if (token_number <= lengh_sent):
                new_token_list.append(new_token) 
                token_number = token_number +1
                listToStr = ' '.join([str(elem) for elem in new_token_list])
        print("\n\n\nFinal result Non_Words: ", listToStr, "\n\n" )
        result,REALWORD = N_Gram_Comparission(listToStr,dictionary,abbreviations )
        #print("\n\n\nFinal Final result Non_Words: ", result, "\n\n" )
        print("\n\n\nFinal Final result REAL WORD: ", REALWORD, "\n\n" )

        #to highlight realword errors in blue
        user_input1.tag_config("blue_tag", foreground="blue")
        for err in REALWORD:
            offset = '+%dc' % len(err)
            pos_start = user_input1.search(err, '1.0', END)
            while pos_start:
                pos_end = pos_start + offset
                user_input1.tag_add("blue_tag", pos_start, pos_end)
                pos_start = user_input1.search(err, pos_end, END)
        break

#function for the clear button
def clear1():
    user_input1.delete(1.0, END)

#search function
def Search():
    
    us = user_search.get()
    if us in dictionary2:
        ind = dictionary2.index(us)
        Diction.selection_set(ind)
        Diction.see(ind)
    else:
        pass

#function to add selected word to dictionary
def add_to_dict(word):

    if word.isalpha():
        dictionary2.append(word)
        dictionary.append(word)
        Diction.insert(END, word)
        print("Added Successfully")
    else:
        print("Please Try Again.")

#defining PDF file source, and calling the read file function
path_to_pdf = 'dmb.pdf'
dictionary, abbreviations = get_pdf_file_content(path_to_pdf)

#building the dictionary consisting of unique tokens, sorted alphabetically
dictionary2 = set(dictionary)
dictionary2 = sorted(dictionary2, key=str.lower)

print("dictionary is:", dictionary2)
print("\n\ntype of dictionary is", type (dictionary2))



#canvas for everything else
HEIGHT = 1200
WIDTH = 1200
canvas = Canvas(height=HEIGHT, width=WIDTH)
canvas.pack()
frame = Frame(bg='#D1C3FF', bd=2)  # this is the gray frame
frame.place(relx=0.5, rely=0.02, relwidth=0.95, relheight=0.95, anchor='n')
#prompting users to input text
main_label1 = Label(frame, text="Enter your words here: ", bg='#D1C3FF',
                    fg="#324390", font="Arial 16 bold")
main_label1.place(relx=0.15, rely=0.01)

#Big scrolling text box for user input
user_input1 = scroll.ScrolledText(frame, width=40, font=("Helvetica", 20),
                                  selectforeground="black", undo=True)
user_input1.focus()
user_input1.pack(expand=True, fill='both')
user_input1.place(relx=0.04, rely=0.05, relwidth=0.4, relheight=0.4)
#clear and enter button
clear_button = Button(frame, text="Clear Screen", width=10, command=clear1)
clear_button.place(relx=0.12, rely=0.455)

Enter_button = Button(frame, text="Enter", width=8, command=main_Function)
Enter_button.place(relx=0.22, rely=0.455)
#second box to store original input
main_label2 = Label(frame, text="User Input:", bg='#D1C3FF',
                    fg="#324390", font="Arial 16 bold")
main_label2.place(relx=0.7, rely=0.01)
user_input2 = scroll.ScrolledText(frame, bg="white", width=50, font="Arial 20")
user_input2.pack(expand=True, fill='both')
user_input2.place(relx=0.53, rely=0.05, relwidth=0.4, relheight=0.4)

# dictionary label
dict_label = Label(frame, text="Dictionary:", bg='#D1C3FF',
                  fg="#324390", font="Arial 16 bold")
dict_label.place(relx=0.2, rely=0.50)

#dictionary box
Diction = Listbox(frame, bg='#FFFFFF', fg="black", font="Arial 20 normal")
for word in dictionary2:
   Diction.insert(END, word)
# Scrollbar for the dictionary list
DicScroll = Scrollbar(Diction, orient=VERTICAL)
DicScroll.config(command=Diction.yview)
DicScroll.pack(side=RIGHT, fill=Y)
# Placing the dictionary into frame
Diction.pack(expand=True, fill='both')
Diction.config(yscrollcommand=DicScroll.set)
Diction.place(relx=0.04, rely=0.54, relwidth=0.4, relheight=0.35)


#Search box
user_search = StringVar()
searchbox = Entry(frame, textvariable=user_search)
searchbox.place(relx=0.04, rely=0.9, relwidth=0.23)
#search button
search_butt = Button(frame, text="Search", width=9, command=Search)
search_butt.place(relx=0.32, rely=0.9)

#Right click popup menu
popup_menu = Menu(root, tearoff=False, background='#1c1b1a',
                  fg='black', activebackground='#534c5c',
                  activeforeground='Yellow')
user_input1.tag_bind("sel", '<Button-2>', popup)


# Writing instructions for the user to follow
instructions1 = Label(frame, text="Instructions for use:", anchor='w', bg='#D1C3FF',
                 fg="#324390", font="Arial 20 bold")
instructions1.place(relx=0.53, rely=0.54, relwidth=0.37)
instructions2 = Label(frame, text="1. Enter your text in the input box.", background='#D1C3FF', fg="#324390",
                 anchor='w',font="Arial 16 normal")
instructions2.place(relx=0.53, rely=0.63, relwidth=0.37)
instructions3 = Label(frame, text="2. Press the 'Enter' button.", background='#D1C3FF', fg="#324390",
                 anchor='w',font="Arial 16 normal")
instructions3.place(relx=0.53, rely=0.66, relwidth=0.37)
instructions4 = Label(frame, text="3. Double-click a highlighted red word to select it, ", fg="#324390",
                      background='#D1C3FF', anchor='w', font="Arial 16 normal")
instructions4.place(relx=0.53, rely=0.69, relwidth=0.37)
instructions4 = Label(frame, text="   then right click to show a list of suggested words.", fg="#324390",
                      background='#D1C3FF', anchor='w',font="Arial 16 normal")
instructions4.place(relx=0.53, rely=0.72, relwidth=0.37)
instructions5 = Label(frame, text="4. Choose a word to replace the error, or ", background='#D1C3FF', fg="#324390",
                 anchor='w',font="Arial 16 normal")
instructions5.place(relx=0.53, rely=0.75, relwidth=0.37)
instructions6 = Label(frame, text="   add the selected word to dictionary.", background='#D1C3FF', fg="#324390",
                 anchor='w',font="Arial 16 normal")
instructions6.place(relx=0.53, rely=0.78, relwidth=0.37)
instructions7 = Label(frame, text="5. Real-word errors are highlighted blue.", background='#D1C3FF', fg="#324390",
                 anchor='w',font="Arial 16 normal")
instructions7.place(relx=0.53, rely=0.81, relwidth=0.37)

root.mainloop()

'''
                            NOTE: keep calm and use stack overflow and git 😁😁😁😁


real-word and Non-real word should have Bigram after check the seplling :
unigram - Bigram - trigram 
"Noisy channel"
decoder
stupid backoff

# idea : first we do non-word spellcorection. after correct all the token we split the doc to sentences
then we find most sentences similar to our sentens. the we do n-gram.
Translate -> then spellcoriction 

prephraze the sentence 

user Correction "List"

GUI
'''


'''
                            NOTE: keep calm and use stack overflow and git 😁😁😁😁

https://stackoverflow.com/questions/1249388/removing-all-non-numeric-characters-from-string-in-python
https://stackoverflow.com/questions/12851791/removing-numbers-from-string
https://stackoverflow.com/questions/4355201/creating-acronyms-in-python
https://stackoverflow.com/questions/55484780/typeerror-argument-of-type-spacy-tokens-token-token-is-not-iterable
https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
https://www.w3resource.com/python-exercises/re/python-re-exercise-42.php
https://stackoverflow.com/questions/17681670/extract-email-sub-strings-from-large-document/176819023
https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/
https://github.com/zoumdatascience/Natural-Language-Processing/blob/master/getPDFContent.ipynb
https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
https://stackoverflow.com/questions/53664775/how-to-remove-punctuation-in-python
https://www.journaldev.com/23763/python-remove-spaces-from-string

https://www.youtube.com/playlist?list=PLJ39kWiJXSixZdkHGyNVwZo8Z3QCm3mx4
https://www.youtube.com/playlist?list=PLJ39kWiJXSizrWpC7hcu1_mLNxEPzN0gF
https://www.youtube.com/playlist?list=PLJ39kWiJXSiz1LK8d_fyxb7FTn4mBYOsD
https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL
https://www.youtube.com/playlist?list=PLc2rvfiptPSToz3K_ozo7zrMJXqe16YUd
https://www.youtube.com/playlist?list=PLc2rvfiptPSQgsORc7iuv7UxhbRJox-pW
https://www.youtube.com/playlist?list=PLBSCvBlTOLa_wS8iy84DfyizdSs7ps7L5
https://www.youtube.com/playlist?list=PLmPJQXJiMoUUSqSV7jcqGiiypGmQ_ogtb
https://www.youtube.com/watch?v=co_cIqGzDxI&ab_channel=eloisa
https://www.youtube.com/watch?v=4yOKlWZk52M&ab_channel=EdwardPie


https://blog.finxter.com/n-grams-in-python-how-they-work/
https://github.com/topics/ngrams

'''


'''
While web analytics allows you to have your finger on the pulse of your website, to monitor the beating 
heart of your online audience, you can easily get 
overwhelmed by the sheer volume of information available to you.
'''

