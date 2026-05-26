import string
import sys
import re
import math

def analyze_string(s):
    def is_prime(n):
        if n<=1 or (n%2==0 and n>2):
            return False
        return all(n%i for i in range(3,int(math.sqrt(n))+1,2))
    vowels="aeiouAEIOU"
    consonants="bcdghjklmnpqrstvwxyzBCDGHJKLMNPQRSTVWXYZ"
    vowel_count=sum(1 for char in s if char.lower() in vowels)
    consonant_count=sum(1 for char in s if char.lower() in consonants)
    digit_count = sum(1 for char in s if char.isdigit())
    unique_chars = len(set(s))
    uppercase_count = sum(1 for char in s if char.isupper())
    Lowercase_count = sum(1 for char in s if char.islower())
    case_ratio=uppercase_count/Lowercase_count if Lowercase_count!=0 else -1
    
    is_palindrome=s=s[::-1]
    numeric_sequence =bool(re.search(r'\d+',s))
    special_count = sum(1 for char in s if not char.isalnum())
    has_special = any(char in string.punctuation for char in s)
    has_digit = any(char.isdigit for char in s)
    has_alphabet = any(char.isalpha()for char in s)
    has_whitespace = any(char.isspace() for char in s)
    has_vowel = any(char in vowels for char in s)
    has_consonant = any(char in consonants for char in s)
    has_uppercase = any(char.isupper()for char in s)
    has_lowercase =any(char.islower()for char in s)
    length = len(s)
    length_is_prime =is_prime(len(s))
    
    score=0
    score+=length==20
    score+=length_is_prime==False
    score+=has_uppercase==True
    score+=has_lowercase==True
    score+=has_digit==True
    score+=has_special==True
    score+=has_vowel==True
    score+=has_consonant==True
    score+=has_whitespace==False
    score+=unique_chars==17
    score+=is_palindrome==False
    score+=has_alphabet==True
    score+=uppercase_count==12
    score+=Lowercase_count==7
    score+=case_ratio==1.7142857142857142
    score+=numeric_sequence==True
    score+=vowel_count==4
    score+=consonant_count==15
    score+=digit_count==1
    score+=special_count==0
    total_case=20
    return score/total_case

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python temp1.py <string>")
        sys.exit(1)
    input_string = sys.argv[1]
    result = analyze_string(input_string)
    print(f"Score: {result:.2f}")
    