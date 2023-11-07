def deciaml_to_binary(decimal):
    binary = ""
    idecimal = int(decimal)
    decimal -= idecimal
    b = 0
    i = 1
    while idecimal > 0:
        b = b + (idecimal%2)*i
        idecimal//=2
        i*=10
    binary = str(b)
    if decimal != 0:
        binary += "."
    while decimal != 0:
        decimal *= 2
        bit = int(decimal)
        binary += str(bit)
        decimal -= bit
    return binary
d = 5.75
b = deciaml_to_binary(d)
print(b)