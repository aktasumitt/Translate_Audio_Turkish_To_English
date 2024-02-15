def Create_Text(want_input:bool=True,want_text:bool=True,text:str=None):
    
    if want_input==True and want_text==False:
        
        text_input=str(input("Please write your task \n(Please put dot(.) between and at the end of the sentences because we will seperated sentences by these dots.)\n"))
        text_list=[]
        print("\n\nCreate text list from input...")
        for i in text_input.split("."):
            text_list.append(i)
        return text_list[:-1]
    
    elif want_text==True and want_input==False:
            print("\n\nCreate text list from your text file...")
            text_list=[]
            for i in text.split("."):
                text_list.append(i)
            return text_list[:-1]
    
    elif want_input==True and want_text==True:
        return (" Please choose just one. Please give text or input.")
    
    else:
        return ("You dont give input or text")

