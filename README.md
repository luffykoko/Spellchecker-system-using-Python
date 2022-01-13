# Spellchecker-system-using-Python

A simple spellchecking system to detect non-word and real-word errors, and providing word suggestions based on the minimal edit distance between the input word and the words in the corpus. 

Here are some screenshots of the spellchecker being used: 

![Picture 2](https://user-images.githubusercontent.com/22349397/149358869-6295a16c-fb10-4dc9-91f5-f7a7b27bc40f.png)
From the screenshot above, the wrong words or 'typos' are highlighted in red, and right-clicking the error word shows a list of suggested words based on the corpus, in order of their minimal edit distance. 


![Picture 3](https://user-images.githubusercontent.com/22349397/149359349-ca0aa050-0c4e-4efc-961c-936e7f834e85.png)
Upon selecting a word from the list, the wrong word is now corrected. 


![Picture 4](https://user-images.githubusercontent.com/22349397/149359464-cac85114-98ed-4848-b10b-0d0a25f3be94.png)
If the input word is not found in our corpus, users can add it into the dictionary list, and it will not be detected as an error thereafter. 

**Credits**
This project was the collective effort of my classmates in the MSc Data Science program, and the job distribution were as follows:

Ms. Lam Ying Xian was our group leader, who was responsible for majority of the literature research work that gave us an idea of which libraries to use, how we should be implementing the edit distances and the n-gram models. She sourced the document which is a digital marketing textbook used as our corpus. SHe also delegated the roles to each of us and constantly monitored the work progress, as well as implementing various versions of the code and testing the spellchecking system together with me and Mr. Rakan Bani Melhem.

Mr. Rakan Bani Melhem wrote majority of the backend code as well as the N-gram model, whilst I wrote the implementation of minimal edit distance and the front-end GUI codes. I was also responsible for streamlining the final code to integrate well with the front-end GUI. 

Mr. Thines Kumar and Mr. Adnan Islam were in charged of formulating the programe design flowchart, as well completion of the report to be submitted as a group assignment, and making sure no errors in spelling and grammer were present, as well as maintaining report aesthetics and formatting. 

