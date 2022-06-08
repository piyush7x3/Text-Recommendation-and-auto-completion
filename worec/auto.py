from fast_autocomplete import AutoComplete
from sympy import content

file1 = open("word.txt","r+") 
s = file1.read()
# Python code to convert string to list
def Convert(string):
    li = list(string.split(" "))
    return li
s = Convert(s)
Words = { stu : {} for stu in s} 
autocomplete = AutoComplete(words=Words)