import re
def verify(string):
    pattern = '(^\d{15}$)|(^\d{17}([0-9]|x)$)'
    match = re.match(pattern, string)
    return match
string = "310101100304983431"
print(verify(string))
