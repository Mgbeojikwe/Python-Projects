"""
1) This source code generates a command line version of the Tic-Tac_Toe game.
2) Four classes(Board, HumanPalyer, ComputerPlayer, TicTacToe) were creted with Baord being the base class, HumanPlayer and ComputerPlayer being 
the child class and class TicTacToe being the grand-child class
3)The check() method of TicTacToe class, has the operations that determines the winner. NumPy fucntion were used in this method
4) the entire game was called with a single method:  game()

"""



import random
import numpy as np
from itertools import chain

class WrongPosition(Exception):pass


class Board:

    positions=[ str(i) for i in range(1,10)]  #positions made a class attributes so that changes effected 
                                            #in HumanPlayer will be seen by ComputerPlayer and vice versa


    def __init__ (self):

        
        self.table=[str(i)  for i in range(1,10)]
        
    def print_board(self):
      for row in  [self.table[i*3:(i+1)*3]  for i in range(3)]:
         print("|".join(row))


class Human_player(Board):


    def __init__(self,letter_H=None):
        Board.__init__(self)
        self.letter_H=letter_H
        
        self.position=None

   
    def play(self):
        
            
            while True:   
                try:
                    # self.position=input(f"select a postion in {self.positions}")
                    self.position=input(f"select a postion in {Board.positions}")
                    if self.position not in Board.positions:
                        raise WrongPosition()
                except WrongPosition:
                    print("position chosen is unavialable") 
                else:
                    break
            Board.positions.remove(self.position) #removing the  position   chosen by the Human Player
            

class ComputerPlayer(Board):

    def __init__(self,letter_C=None):
        Board.__init__(self)
        self.letter_C=letter_C
        self.position=None

        
    def play(self):

        self.position=random.choice(self.positions)

        Board.positions.remove(self.position)    #removing the position chosen by the Computer       
        #print(se.positions)
        print(f"computer choose {self.position}")

#creating objects for the classes
Human=Human_player('X')
Computer=ComputerPlayer('O')

"""
............................................................................................
"""


#Creating a class that inherits both the HumanPlayer and Computer Player classes

class TicTacToe(Human_player,ComputerPlayer):


    def __init__(self, letter_H=None, letter_C=None, letter_T=None):
    
        Human_player.__init__(self,letter_H)
        ComputerPlayer.__init__(self,letter_C)
        self.letter_T = letter_T

        pass

    
    def check(self):
        """ 
        1) This methods checks knows if a winner has emerged at the current state of the game
        2)It starts by converting "self.table" into a (3X3) matrix "row_table1 followed by taking the inverse.
            Then checks if elemnts of a given row are of same type 'X' or 'O'. If true it retures "True"
            and ends the fucntion 
        3) If step2 is false, then function now checks if the elements are the diagonals are same
            If same, it returns "True", thus existing the loop
        """
        row_table1 = np.array([self.table[i*3:(i+1)*3]  for i in range(3)]) 
        row_table2 = row_table1.T
        row_tables = [row_table1,row_table2]  #forming a list of two row_tables


        for row_table in row_tables:

            for row in row_table:
                if all([element=='X'  for element in row])==True:
                    print("you won")
                    return True 
                if all([element=='O' for element in row])==True:
                    print("You losed")
                    return True 
        
        if all([self.table[i]=='X'   for i in [0,4,8]])==True or all([self.table[i]=='X' for i in [2,4,6]])==True:
            print("You won")
            return True 
        if all([self.table[i]=='O'   for i in [0,4,8]])==True or all([self.table[i]=='O' for i in [2,4,6]])==True:
            print("you losed")
            return True

        if len(Board.positions) <= 1:
            print("It is a draw")
            return True


    def game(self):
        
        result=False
        
        while (len(Board.positions)>0 and result !=True  ):
        

            Human.play()
            Computer.play()
            
            
            
            #assigning the respective letters to their positions
            for index,position in enumerate(self.table):
                if position==Human.position:
                    # self.table[index] = Human.letter
                    self.table[index] = Human.letter_H
                if position == Computer.position:
                    # self.table[index]=Computer.letter
                    self.table[index]=Computer.letter_C
            
            self.print_board() #printing out the curretn form of the table
            result=self.check()

        if result== False:
            print("It is a tie")





obj=TicTacToe()
print("In this game you are the first player while computer is the seconf player")
print(f"In this game you will use {Human.letter_H} while computer will use {Computer.letter_C}\n")
obj.game()

